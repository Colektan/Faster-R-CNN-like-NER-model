import torch.nn as nn
import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size  # 768
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        """
        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)  # (seq_length,batch_size,num_directions*hidden_size)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)  # [seq_length, batch_size, num_labels]
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())

class FRLN(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        self.IoU_threshold = 0.75

        super(FRLN, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size  # 768

        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.multi_loss_fn = nn .CrossEntropyLoss(ignore_index=0)

        self.need_birnn = need_birnn
        if need_birnn:
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.first_classification = nn.Linear(out_dim, 1) # predict is an entity or not
        self.loc_regression = nn.Linear(out_dim, 2) # predict possible word locations

        self.classification = nn.Linear(out_dim, 4) #predict the type of the entity

    def forward(self, input_ids, tags, word_areas, total_areas, total_tags, token_type_ids=None, attention_mask=None):
        """
        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])

        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)  # (seq_length,batch_size,num_directions*hidden_size)

        cls_output = self.first_classification(sequence_output) # [batch_size, seq_len, 1]
        loc_output = self.loc_regression(sequence_output) # [batch_size, seq_len, 2]

        temp_loss = self.binary_loss_fn(cls_output.squeeze(-1), word_areas[:, :, 0])
        cls_loss = (attention_mask * self.binary_loss_fn(cls_output.squeeze(-1), word_areas[:, :, 0])).sum(axis=-1) / attention_mask.sum(axis=-1)
        distance = (loc_output - word_areas[:, :, 1:]) * (loc_output - word_areas[:, :, 1:])
        loc_loss = (attention_mask * word_areas[:, :, 0] * (distance).sum(axis=-1)).sum(axis=-1) / attention_mask.sum(axis=-1)

        cls_loss = cls_loss.mean()

        loc_loss = loc_loss.mean()

        merged_output =  (sequence_output.unsqueeze(1) * total_areas.unsqueeze(-1)).sum(axis=-2)  # [batch, seq_length, dimension]

        final_cls_output = self.classification(merged_output)

        final_cls_loss = self.multi_loss_fn(final_cls_output.permute(0, 2, 1), total_tags)

        return cls_loss, loc_loss, final_cls_loss
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)  # (seq_length,batch_size,num_directions*hidden_size)
        
        cls_output = self.first_classification(sequence_output)
        loc_output = self.loc_regression(sequence_output) # [batch_size, seq_len, 2]

        cls_mask = nn.functional.sigmoid(cls_output).squeeze(-1)
        cls_mask = torch.where(cls_mask > 0.5, 1, 0)
        cls_mask = cls_mask * attention_mask

        base_loc = torch.arange(128).reshape(-1, 1).tile(2).cuda()

        # round and filter invalid values
        absolute_loc = base_loc + loc_output
        absolute_loc = torch.clip(absolute_loc, 0, 127)

        loc_masks = torch.zeros((absolute_loc.shape[0], absolute_loc.shape[1], absolute_loc.shape[1])).cuda() # [batch_size, seq_len, seq_len]

        # filter proposed region based on IoU
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                if cls_mask[row, col] == 0:
                    continue
                for idx in range(col + 1, absolute_loc.shape[1]):
                    if cls_mask[row, idx] == 0:
                        continue
                    # if IoU > threshold, drop the second one by setting the mask to zero.
                    pos1_left = absolute_loc[row, col, 0]
                    pos1_right = absolute_loc[row, col, 1]
                    pos2_left = absolute_loc[row, idx, 0]
                    pos2_right = absolute_loc[row, idx, 1]

                    if pos1_right < pos2_left:
                        continue
                    elif pos2_left > pos1_left:
                        if pos2_right <= pos1_right:
                            cls_mask[row, idx] = 0
                        elif (pos1_right - pos2_left + 1) / (pos2_right - pos1_left + 1) >= self.IoU_threshold:
                            cls_mask[row, idx] = 0
                        else:
                            absolute_loc[row, col, 1] = pos2_left - 1
                    elif pos2_right >= pos1_right:
                        cls_mask[row, col] = 0
                    elif pos2_right >= pos1_left:
                        if (pos2_right - pos1_left + 1) / (pos1_right - pos2_left + 1) >= self.IoU_threshold:
                            cls_mask[row, col] = 0
                        else:
                            absolute_loc[row, idx, 1] = pos1_left - 1
                    else:
                        continue

                    # if pos1_right < pos2_left:
                    #     continue
                    # elif pos2_left > pos1_left:
                    #     if pos2_right <= pos1_right:
                    #         cls_mask[row, idx] = 0
                    #         continue
                    #     else:
                    #         cls_mask[row, col] = 0
                    #         absolute_loc[row, idx, 0] = pos1_left
                    #         break
                    # elif pos2_right >= pos1_right:
                    #     cls_mask[row, col] = 0
                    #     break
                    # elif pos2_right >= pos1_left:
                    #     cls_mask[row, col] = 0
                    #     absolute_loc[row, idx, 1] = pos1_right
                    #     break
                    # else:
                    #     continue
                    

        absolute_loc = torch.round(absolute_loc)
        absolute_loc = absolute_loc.int()
                    
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                loc_masks[row, col, absolute_loc[row, col, 0] : absolute_loc[row, col, 1] + 1] += 1 / (absolute_loc[row, col, 1] - absolute_loc[row, col, 0] + 1)

        cls_input = cls_mask[:, :, None, None] * loc_masks[:, :, :, None] * sequence_output[:, None, :, :]
        cls_input = cls_input.sum(axis=-2) # [batch, seq, 768]

        cls_output = self.classification(cls_input) #[batch, seq, 4]
        tag =  torch.argmax(cls_output, dim=-1)
        first_tag = [4, 6, 3]
        second_tag = [0, 1, 2]

        # generate predictions
        logits = np.zeros((absolute_loc.shape[0], absolute_loc.shape[1])).astype('int') + 5
        logits[:, 0] = 7
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                if cls_mask[row, col] == 0 or tag[row, col] == 0:
                    continue
                logits[row, absolute_loc[row, col, 0]] = first_tag[tag[row, col] - 1]
                logits[row, absolute_loc[row, col, 0] + 1 : absolute_loc[row, col, 1] + 1] = second_tag[tag[row, col] - 1]
        
        output = []
        for row in range(absolute_loc.shape[0]):
            temp = []
            for col in range(absolute_loc.shape[1]):
                if attention_mask[row, col] == 0:
                    temp[-1] = 8
                    break
                temp.append(int(logits[row, col]))
            output.append(temp)
        return output

class FRLNforClueNER(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        self.IoU_threshold = 0.9

        super(FRLNforClueNER, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size  # 768

        self.first_classification = nn.Linear(out_dim, 1) # predict is an entity or not
        self.loc_regression = nn.Linear(out_dim, 2) # predict possible word locations

        self.classification = nn.Linear(out_dim, 11) #predict the type of the entity

        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.multi_loss_fn = nn .CrossEntropyLoss(ignore_index=0)

        if need_birnn:
            self.need_birnn = need_birnn
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, word_areas, total_areas, total_tags, token_type_ids=None, attention_mask=None):
        """
        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])

        cls_output = self.first_classification(sequence_output) # [batch_size, seq_len, 1]
        loc_output = self.loc_regression(sequence_output) # [batch_size, seq_len, 2]

        temp_loss = self.binary_loss_fn(cls_output.squeeze(-1), word_areas[:, :, 0])
        cls_loss = (attention_mask * self.binary_loss_fn(cls_output.squeeze(-1), word_areas[:, :, 0])).sum(axis=-1) / attention_mask.sum(axis=-1)
        distance = (loc_output - word_areas[:, :, 1:]) * (loc_output - word_areas[:, :, 1:])
        loc_loss = (attention_mask * word_areas[:, :, 0] * (distance).sum(axis=-1)).sum(axis=-1) / attention_mask.sum(axis=-1)

        cls_loss = cls_loss.mean()

        loc_loss = loc_loss.mean()

        merged_output =  (sequence_output.unsqueeze(1) * total_areas.unsqueeze(-1)).sum(axis=-2)  # [batch, seq_length, dimension]

        final_cls_output = self.classification(merged_output)

        final_cls_loss = self.multi_loss_fn(final_cls_output.permute(0, 2, 1), total_tags)

        return cls_loss, loc_loss, final_cls_loss
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        cls_output = self.first_classification(sequence_output)
        loc_output = self.loc_regression(sequence_output) # [batch_size, seq_len, 2]

        cls_mask = nn.functional.sigmoid(cls_output).squeeze(-1)
        cls_mask = torch.where(cls_mask > 0.5, 1, 0)
        cls_mask = cls_mask * attention_mask

        base_loc = torch.arange(128).reshape(-1, 1).tile(2).cuda()

        # round and filter invalid values
        absolute_loc = base_loc + loc_output
        absolute_loc = torch.clip(absolute_loc, 0, 127)

        loc_masks = torch.zeros((absolute_loc.shape[0], absolute_loc.shape[1], absolute_loc.shape[1])).cuda() # [batch_size, seq_len, seq_len]

        # filter proposed region based on IoU
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                if cls_mask[row, col] == 0:
                    continue
                for idx in range(col + 1, absolute_loc.shape[1]):
                    if cls_mask[row, idx] == 0:
                        continue
                    # if IoU > threshold, drop the second one by setting the mask to zero.
                    pos1_left = absolute_loc[row, col, 0]
                    pos1_right = absolute_loc[row, col, 1]
                    pos2_left = absolute_loc[row, idx, 0]
                    pos2_right = absolute_loc[row, idx, 1]

                    # if pos1_right < pos2_left:
                    #     continue
                    # elif pos2_left > pos1_left:
                    #     if pos2_right <= pos1_right:
                    #         cls_mask[row, idx] = 0
                    #     elif (pos1_right - pos2_left + 1) / (pos2_right - pos1_left + 1) >= self.IoU_threshold:
                    #         cls_mask[row, idx] = 0
                    #     else:
                    #         absolute_loc[row, col, 1] = pos2_left - 1
                    # elif pos2_right >= pos1_right:
                    #     cls_mask[row, col] = 0
                    # elif pos2_right >= pos1_left:
                    #     if (pos2_right - pos1_left + 1) / (pos1_right - pos2_left + 1) >= self.IoU_threshold:
                    #         cls_mask[row, col] = 0
                    #     else:
                    #         absolute_loc[row, idx, 1] = pos1_left - 1
                    # else:
                    #     continue

                    if pos1_right < pos2_left:
                        continue
                    elif pos2_left > pos1_left:
                        if pos2_right <= pos1_right:
                            cls_mask[row, idx] = 0
                            continue
                        else:
                            cls_mask[row, col] = 0
                            absolute_loc[row, idx, 0] = pos1_left
                            break
                    elif pos2_right >= pos1_right:
                        cls_mask[row, col] = 0
                        break
                    elif pos2_right >= pos1_left:
                        cls_mask[row, col] = 0
                        absolute_loc[row, idx, 1] = pos1_right
                        break
                    else:
                        continue
                    

        absolute_loc = torch.round(absolute_loc)
        absolute_loc = absolute_loc.int()
                    
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                loc_masks[row, col, absolute_loc[row, col, 0] : absolute_loc[row, col, 1] + 1] += 1 / (absolute_loc[row, col, 1] - absolute_loc[row, col, 0] + 1)

        cls_input = cls_mask[:, :, None, None] * loc_masks[:, :, :, None] * sequence_output[:, None, :, :]
        cls_input = cls_input.sum(axis=-2) # [batch, seq, 768]

        cls_output = self.classification(cls_input) #[batch, seq, 4]
        tag =  torch.argmax(cls_output, dim=-1)
        first_tag = [4, 6, 3]
        second_tag = [0, 1, 2]

        # generate predictions
        logits = np.zeros((absolute_loc.shape[0], absolute_loc.shape[1])).astype('int') + 5
        logits[:, 0] = 7
        for row in range(absolute_loc.shape[0]):
            for col in range(absolute_loc.shape[1]):
                if cls_mask[row, col] == 0 or tag[row, col] == 0:
                    continue
                logits[row, absolute_loc[row, col, 0]] = first_tag[tag[row, col] - 1]
                logits[row, absolute_loc[row, col, 0] + 1 : absolute_loc[row, col, 1] + 1] = second_tag[tag[row, col] - 1]
        
        output = []
        for row in range(absolute_loc.shape[0]):
            temp = []
            for col in range(absolute_loc.shape[1]):
                if attention_mask[row, col] == 0:
                    temp[-1] = 8
                    break
                temp.append(int(logits[row, col]))
            output.append(temp)
        return output
