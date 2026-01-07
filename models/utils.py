import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#from bidirectional_cross_attention import BidirectionalCrossAttention
from .cbam import CBAM

def resize_image(image_tensor, new_height, new_width):
    # Reshape the image tensor to [batch_size * channels, height, width]
    batch_size, channels, height, width = image_tensor.size()
    reshaped_image_tensor = image_tensor.view(-1, channels, height, width)

    # Resize the image using bilinear interpolation
    resized_image_tensor = F.interpolate(reshaped_image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Reshape the resized image tensor back to its original shape
    resized_image_tensor = resized_image_tensor.view(batch_size, channels, new_height, new_width)

    return resized_image_tensor

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)

        return x

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values):
        """
        Args:
            queries (torch.Tensor): Tensor of shape (batch_size, num_queries, embedding_dim).
            keys (torch.Tensor): Tensor of shape (batch_size, num_keys, embedding_dim).
            values (torch.Tensor): Tensor of shape (batch_size, num_keys, embedding_dim).

        Returns:
            torch.Tensor: Weighted sum of values, with attention weights applied.
        """
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))  # (batch_size, num_queries, num_keys)
        
        # Normalize attention scores with softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_queries, num_keys)
        
        # Apply attention weights to values
        weighted_sum = torch.matmul(attention_weights, values)  # (batch_size, num_queries, embedding_dim)
        
        return weighted_sum

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        #self.e1 = encoder_block(10, 256)
        #self.e2 = encoder_block(64, 128)
        #self.e3 = encoder_block(128, 256)
        #self.e4 = encoder_block(256, 512)

        self.e1 = encoder_block(10, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        #b = self.b(p4)

        """ Decoder """
        #d1 = self.d1(b, s4)
        #d2 = self.d2(d1, s3)
        #d3 = self.d3(d2, s2)
       # d4 = self.d4(d3, s1)

        """ Classifier """
        #outputs = self.outputs(d4)

        return s4

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


class VL_block(nn.Module):
    def __init__(self, hidden_dim, key_value_dimension):
        super(VL_block, self).__init__()
        # Linear function

        self.hidden_dim = hidden_dim
        self.key_value_dimension = key_value_dimension

        self.self_att = nn.MultiheadAttention(512, 4, 0.10, batch_first=True)
        self.cross_att = nn.MultiheadAttention(512, 4, 0.10, batch_first=True)
        
        self.depth = 3
        self.query_transform = nn.Linear(120*160, hidden_dim)
        self.key_transform = nn.Linear(key_value_dimension, hidden_dim)
        self.value_transform = nn.Linear(key_value_dimension, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 120*160), nn.ReLU(True))

    def forward(self, img, txt):
        keys = self.key_transform(txt)       
        values = self.value_transform(txt) 

        img = img.view(2, 256, -1)
        img = self.query_transform(img)

        for depth in range(self.depth):
            weighted_values, _ = self.self_att(img, img, img)
            weighted_values, _ = self.cross_att(weighted_values, keys, values)

        out = self.classifier(weighted_values)
        out = out.view(2, 256, 120, 160)

        return out

class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        #scores = scores.masked_fill(mask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=512, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model=1024,
        heads=8, 
        feed_forward_hidden=1024 * 8, 
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, queries, keys_values):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(queries, keys_values, keys_values))
        embeddings = queries
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class VisionLanguageFusion(torch.nn.Module):
    def __init__(
        self,
    ):
        super(VisionLanguageFusion, self).__init__()
        self.projector_img = nn.Linear(19200, 1024)
        self.projector_img2 = nn.Linear(19200, 1024)
        self.projector_txt = nn.Linear(256, 1024)
        self.projector_txt2 = nn.Linear(256, 1024)
        self.projector_txt3 = nn.Linear(19200, 1024)
        self.projector_txt4 = nn.Linear(19200, 1024)
        #self.block_sem = EncoderLayer()
        #self.block_dep = EncoderLayer()
        self.projector_back = nn.Linear(1024, 19200)
        self.projector_back2 = nn.Linear(1024, 19200)
        self.num_layers = 4
        self.sem_layers = nn.ModuleList()
        self.dep_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer()
            dep_layer = EncoderLayer()
            self.sem_layers.append(layer)
            self.dep_layers.append(dep_layer)

    def forward(self, sem_text, depth_text, sem_img, dep_img):
        # Flatten Image Representations
        sem_img = sem_img.view(sem_img.shape[0], sem_img.shape[1], -1)
        dep_img = dep_img.view(dep_img.shape[0], dep_img.shape[1], -1)

        # Linear Transformations
        sem_img_ = self.projector_img(sem_img)
        dep_img_ = self.projector_img2(dep_img)
        if sem_text.shape[-1] == 256:
            sem_txt_ = self.projector_txt(sem_text)
            depth_txt_ = self.projector_txt2(depth_text)
        else:
            sem_txt_ = self.projector_txt3(sem_text)
            depth_txt_ = self.projector_txt4(depth_text)

        # Bert Layer Cross Attention
        for layer in range(self.num_layers):
            sem_layer = self.sem_layers[layer]
            dep_layer = self.dep_layers[layer]
            sem_text_vision = sem_layer(sem_txt_, sem_img_)
            dep_txt_vision = dep_layer(depth_txt_, dep_img_)

        # Linear Transformations
        sem_text_vision = self.projector_back(sem_text_vision)
        dep_txt_vision = self.projector_back2(dep_txt_vision)

        # Concatenation
        sem_concat = torch.cat((sem_img, sem_text_vision), dim=1)
        dep_concat = torch.cat((dep_img, dep_txt_vision), dim=1)

        # Resize
        sem_concat = sem_concat.view(sem_concat.shape[0], sem_concat.shape[1], 120, 160)
        dep_concat = dep_concat.view(dep_concat.shape[0], dep_concat.shape[1], 120, 160)

        return sem_concat, dep_concat, sem_text_vision, dep_txt_vision

    def forward2(self, sem_text, depth_text, shared_img):
        shared_img = shared_img.view(shared_img.shape[0], shared_img.shape[1], -1)

        shared_img_ = self.projector_img(shared_img)
        if sem_text.shape[-1] == 256:
            sem_txt_ = self.projector_txt(sem_text)
            depth_txt_ = self.projector_txt2(depth_text)
        else:
            sem_txt_ = self.projector_txt3(sem_text)
            depth_txt_ = self.projector_txt4(depth_text)

        sem_text_vision = self.block_sem(sem_txt_, shared_img_)
        dep_txt_vision = self.block_dep(depth_txt_, shared_img_)

        sem_text_vision = self.projector_back(sem_text_vision)
        dep_txt_vision = self.projector_back2(dep_txt_vision)

        shared_rep = torch.cat((shared_img, sem_text_vision), dim=1)
        shared_rep = torch.cat((shared_rep, dep_txt_vision), dim=1)

        shared_rep = shared_rep.view(shared_rep.shape[0], shared_rep.shape[1], 120, 160)

        return shared_rep, sem_text_vision, dep_txt_vision


class EncoderLayer2(torch.nn.Module):
    def __init__(
        self, 
        d_model=1024,
        heads=4, 
        feed_forward_hidden=512, 
        dropout=0.1
        ):
        super(EncoderLayer2, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.cross_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, queries, keys_values):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.self_multihead(queries, keys_values, keys_values)
        interacted = self.cross_multihead(interacted, keys_values, keys_values)
        embeddings = queries
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.feed_forward(interacted)
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded

class EncoderLayer3(torch.nn.Module):
    def __init__(
        self, 
        d_model=1024,
        heads=4, 
        feed_forward_hidden=512, 
        dropout=0.1
        ):
        super(EncoderLayer3, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        #self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.cross_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, queries, keys_values):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.cross_multihead(queries, keys_values, keys_values)
        interacted = self.cross_multihead(interacted, interacted, interacted)
        embeddings = queries
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.feed_forward(interacted)
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded

class VisionLanguageFusion2(torch.nn.Module):
    def __init__(
        self,
    ):
        super(VisionLanguageFusion2, self).__init__()
        self.projector_img = nn.Linear(19200, 512)
        self.projector_img2 = nn.Linear(19200, 512)
        self.projector_txt = nn.Linear(1536, 512)
        self.projector_txt2 = nn.Linear(256, 512)
        self.projector_txt3 = nn.Linear(19200, 512)
        self.projector_txt4 = nn.Linear(19200, 512)
        self.projector_back = nn.Linear(512, 19200)
        self.projector_back2 = nn.Linear(512, 19200)
        self.num_layers = 4
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer3()
            self.layers.append(layer)

    def forward(self, img, txt):
        # Flatten Image Representations
        img = img.view(img.shape[0], img.shape[1], -1)
        #breakpoint()

        # Linear Transformations
        img = self.projector_img(img)
        #breakpoint()

        if txt.shape[-1] == 1536:
            txt = self.projector_txt(txt)
        else:
            txt = self.projector_txt3(txt)

        # Bert Layer Cross Attention
        #breakpoint()
        for layer in range(self.num_layers):
            fuse_layer = self.layers[layer]
            rep = fuse_layer(img, txt)

        # Linear Transformations
        rep = self.projector_back(rep)
        #dep = self.projector_back2(dep)
        #breakpoint()
        # Concatenation
        #sem_concat = torch.cat((sem_img, sem_text_vision), dim=1)
        #dep_concat = torch.cat((dep_img, dep_txt_vision), dim=1)

        # Resize
        rep = rep.view(rep.shape[0], rep.shape[1], 120, 160)
        #dep = dep.view(dep.shape[0], dep.shape[1], 120, 160)

        return rep

    
class VisionLanguageFusion3(torch.nn.Module):
    def __init__(
        self,
    ):
        super(VisionLanguageFusion3, self).__init__()
        self.projector_img = nn.Linear(19200, 1024)
        self.projector_img2 = nn.Linear(19200, 1024)
        self.projector_txt = nn.Linear(256, 1024)
        self.projector_txt2 = nn.Linear(256, 1024)
        self.projector_txt3 = nn.Linear(19200, 1024)
        self.projector_txt4 = nn.Linear(19200, 1024)
        self.projector_back = nn.Linear(1024, 19200)
        self.projector_back2 = nn.Linear(1024, 19200)
        self.num_layers = 2
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer2()
            self.layers.append(layer)

    def forward(self, img, txt):
        # Flatten Image Representations
        img = img.view(img.shape[0], img.shape[1], -1)
        #breakpoint()

        # Linear Transformations
        img = self.projector_img(img)
        #breakpoint()

        if txt.shape[-1] == 256:
            txt = self.projector_txt(txt)
        else:
            txt = self.projector_txt3(txt)

        # Bert Layer Cross Attention
        #breakpoint()
        for layer in range(self.num_layers):
            fuse_layer = self.layers[layer]
            rep = fuse_layer(img, txt)

        # Linear Transformations
        rep = self.projector_back(rep)
        #dep = self.projector_back2(dep)
        #breakpoint()
        # Concatenation
        #sem_concat = torch.cat((sem_img, sem_text_vision), dim=1)
        #dep_concat = torch.cat((dep_img, dep_txt_vision), dim=1)

        # Resize
        rep = rep.view(rep.shape[0], rep.shape[1], 120, 160)
        #dep = dep.view(dep.shape[0], dep.shape[1], 120, 160)

        return rep

class VisionLanguageFusion4(torch.nn.Module):
    def __init__(
        self,
    ):
        super(VisionLanguageFusion4, self).__init__()
        self.projector_img = nn.Linear(19200, 1024)
        self.projector_img2 = nn.Linear(19200, 1024)
        self.projector_txt = nn.Linear(256, 1024)
        self.projector_txt2 = nn.Linear(256, 1024)
        self.projector_txt3 = nn.Linear(19200, 1024)
        self.projector_txt4 = nn.Linear(19200, 1024)
        self.projector_back = nn.Linear(1024, 19200)
        self.projector_back2 = nn.Linear(1024, 19200)
        self.num_layers = 4
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer3()
            self.layers.append(layer)

    def forward(self, img, txt):
        # Flatten Image Representations
        img = img.view(img.shape[0], img.shape[1], -1)
        #breakpoint()

        # Linear Transformations
        img = self.projector_img(img)
        #breakpoint()

        if txt.shape[-1] == 256:
            txt = self.projector_txt(txt)
        else:
            txt = self.projector_txt3(txt)

        # Bert Layer Cross Attention
        #breakpoint()
        for layer in range(self.num_layers):
            fuse_layer = self.layers[layer]
            rep = fuse_layer(img, txt)

        # Linear Transformations
        rep = self.projector_back(rep)
        #dep = self.projector_back2(dep)
        #breakpoint()
        # Concatenation
        #sem_concat = torch.cat((sem_img, sem_text_vision), dim=1)
        #dep_concat = torch.cat((dep_img, dep_txt_vision), dim=1)

        # Resize
        rep = rep.view(rep.shape[0], rep.shape[1], 120, 160)
        #dep = dep.view(dep.shape[0], dep.shape[1], 120, 160)

        return rep

""" class Bidirectional_Cross_Attention(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Bidirectional_Cross_Attention, self).__init__()
        self.num_layers = 2
        self.layers = nn.ModuleList()
        self.img_projector = nn.Linear(19200, 256)
        self.img_mask = torch.ones((2, 256)).bool().cuda()
        self.txt_mask = torch.ones((2, 32)).bool().cuda()
        self.projector_back = nn.Linear(256, 19200)
        for i_layer in range(self.num_layers):
            layer = BidirectionalCrossAttention(dim=256, heads = 8, dim_head = 64, context_dim = 256)
            self.layers.append(layer)
            
        
    def forward(self, img, txt):
        img = self.img_projector(img.view(img.shape[0], img.shape[1], -1))
        for layer in range(self.num_layers):
            layer = self.layers[layer]
            img, txt = layer(
                                img,
                                txt,
                                mask = self.img_mask,
                                context_mask = self.txt_mask
            )
        img_out = self.projector_back(img)
        img_out = img_out.view(img.shape[0], img.shape[1], 120, 160)
        return img_out, txt """



class CrossAttention(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(1, output_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(1, output_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)

    def forward(self, segmentation_predictions, depth_map_embeddings):
        """
        Args:
            segmentation_predictions (torch.Tensor): Semantic segmentation predictions with shape (batch_size, num_classes, height, width).
            depth_map_embeddings (torch.Tensor): Depth map embeddings with shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Cross-attended output with the same shape as segmentation_predictions.
        """
        batch_size, num_classes, height, width = segmentation_predictions.size()
        
        # Compute queries, keys, and values
        queries = self.query_conv(segmentation_predictions)  # (batch_size, output_channels, height, width)
        keys = self.key_conv(depth_map_embeddings)  # (batch_size, output_channels, height, width)
        values = self.value_conv(depth_map_embeddings)  # (batch_size, output_channels, height, width)

        # Reshape queries and keys for efficient matrix multiplication
        queries = queries.view(batch_size, -1, height * width)  # (batch_size, output_channels, height*width)
        keys = keys.view(batch_size, -1, height * width)  # (batch_size, output_channels, height*width)
        values = values.view(batch_size, -1, height * width)  # (batch_size, output_channels, height*width)

        # Compute attention scores
        attention_scores = torch.bmm(queries.transpose(1, 2), keys)  # (batch_size, height*width, height*width)

        # Normalize attention scores with softmax
        attention_weights = self.softmax(attention_scores)  # (batch_size, height*width, height*width)

        # Apply attention weights to values
        cross_attended_output = torch.bmm(values, attention_weights.transpose(1, 2))  # (batch_size, output_channels, height*width)

        # Reshape cross-attended output to match the original shape
        cross_attended_output = cross_attended_output.view(batch_size, -1, height, width)  # (batch_size, output_channels, height, width)

        # Apply convolution to combine attended output
        cross_attended_output = self.out_conv(cross_attended_output)  # (batch_size, output_channels, height, width)

        return cross_attended_output

def spread_transformation(data, factor):
    # Extract the values from the dictionary
    values = list(data.values())
    
    # Calculate the mean of the values
    mean = sum(values) / len(values)
    
    # Calculate the deviations from the mean
    deviations = [value - mean for value in values]
    
    # Scale the deviations and compute the new values
    new_values = [mean + factor * deviation for deviation in deviations]
    
    # Create a new dictionary with the same keys but transformed values
    transformed_data = dict(zip(data.keys(), new_values))
    
    return transformed_data


class DotProductFusion(torch.nn.Module):
    def __init__(
        self,
    ):
        super(DotProductFusion, self).__init__()
        self.depth = 2
        self.layers = nn.ModuleList()
        self.layer1 = CrossAttention()
    

    def forward(self, task1, task2):
        # Flatten Image Representations
        output = self.layer1(task1, task2)
        return output


class SPF(torch.nn.Module):
    def __init__(
        self, pred_dim, back_dim
    ):
        super(SPF, self).__init__()
        self.pred_dim = pred_dim
        self.back_dim = back_dim
        self.sem_cbam = CBAM(gate_channels=pred_dim)
        self.dep_cbam = CBAM(gate_channels=back_dim)
        self.sem_conv = nn.Conv2d(pred_dim+back_dim,256,1)
        self.dep_conv = nn.Conv2d(pred_dim+back_dim,back_dim,1)
        self.relu = nn.ReLU()
        self.b1 = nn.BatchNorm2d(256)
        self.b2 = nn.BatchNorm2d(back_dim)

    def forward(self, pred, back_feat):
        #breakpoint()
        pred = self.sem_cbam(pred)#.relu()
        #breakpoint()
        back_feat = self.dep_cbam(back_feat)#.relu()
        #breakpoint()
        pred = resize_image(pred, back_feat.size()[-2], back_feat.size()[-1])
        #breakpoint()
        concatenated_rep = torch.cat((pred, back_feat), dim=1)
        pred = self.relu(self.b1(self.sem_conv(concatenated_rep)))
        back_feat = self.relu(self.b2(self.dep_conv(concatenated_rep)))
        #breakpoint()
        return pred, back_feat


class TraceBack(torch.nn.Module):
    def __init__(
        self,
    ):
        super(TraceBack, self).__init__()
        self.num_layers = 4
        self.spf_modules = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                layer = SPF(256, 1024) # (192, 1024)
            elif i == 1:
                layer = SPF(1024, 512) # (1024, 512)
            elif i == 2:
                layer = SPF(512, 256) # (512, 256)
            self.spf_modules.append(layer)

    def forward(self, liste, initial_pred=None):
        # Flatten Image Representations
        pred_list = []
        for i, feature in enumerate(liste):
            #print(i)
            spf_module = self.spf_modules[i]
            #breakpoint()
            if i ==0:
                pred, back_feat = spf_module(initial_pred, feature)
                pred_list.append(pred)
                #breakpoint()
            else:
                pred, back_feat = spf_module(back_feat, feature)
                pred_list.append(pred)
                #breakpoint()

        return pred_list


class GAAF_module(torch.nn.Module):
    def __init__(
        self, dim
    ):
        super(GAAF_module, self).__init__()
        self.dim = dim
        self.cbam_sem = CBAM(gate_channels=self.dim)
        self.cbam_dep = CBAM(gate_channels=self.dim*2)
        self.cbam_dep2 = CBAM(gate_channels=self.dim*2)
        self.shared_dep2 = CBAM(gate_channels=self.dim)
        self.cbam_sem_norm = nn.BatchNorm2d(self.dim)
        self.cbam_dep_norm = nn.BatchNorm2d(self.dim*2)
        self.cbam_dep2_norm = nn.BatchNorm2d(self.dim*2)
        self.shared_dep3 = CBAM(gate_channels=self.dim)
        self.shared_dep2_norm = nn.BatchNorm2d(self.dim*2)

    def forward(self, sem_embeds, depth_embeds):
        sem = self.cbam_sem(sem_embeds)
        depth1 = self.cbam_dep(depth_embeds)
        depth2 = self.cbam_dep2(depth_embeds)

        batch_size, channels, height, width = depth2.size()
        theta_d=depth1.view(batch_size,channels,-1)
        theta_d=theta_d.permute(0,2,1)
        phi_d=depth2.view(batch_size,channels,-1)
        depth_mix=torch.matmul(theta_d,phi_d)
        depth_mix=F.softmax(depth_mix,dim=-1)

        g_s=sem.view(batch_size, self.dim,-1)
        g_s=g_s.permute(0,2,1)
        out = torch.matmul(depth_mix, g_s)
        out = out.permute(0,2,1).contiguous()

        out = out.view(batch_size, self.dim, height, width)
        out = self.shared_dep2(out) + sem_embeds
        return out

class GAAF_module2(torch.nn.Module):
    def __init__(
        self, dim
    ):
        super(GAAF_module2, self).__init__()
        self.dim = dim
        self.cbam_sem = CBAM(gate_channels=self.dim)
        self.cbam_dep = CBAM(gate_channels=self.dim*2)
        self.cbam_dep2 = CBAM(gate_channels=self.dim*2)
        self.shared_dep2 = CBAM(gate_channels=self.dim)
        self.cbam_sem_norm = nn.BatchNorm2d(self.dim)
        self.cbam_dep_norm = nn.BatchNorm2d(self.dim*2)
        self.cbam_dep2_norm = nn.BatchNorm2d(self.dim*2)
        self.shared_dep2_norm = nn.BatchNorm2d(self.dim)
        self.gate1 = nn.Conv2d(256, 256, kernel_size=1)
        self.gate2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, sem_embeds, depth_embeds):
        sem = self.cbam_sem_norm(self.cbam_sem(sem_embeds))
        depth1, depth2 = depth_embeds[:, :256, :, :], depth_embeds[:, 256:, :, :]
        
        gate1 = torch.sigmoid(self.gate1(depth1))
        gate2 = torch.sigmoid(self.gate2(depth2))
        depth1_weighted = depth1 * gate1
        depth2_weighted = depth2 * gate2
        concatenated = torch.cat([depth1_weighted, depth2_weighted], dim=1)
        fused = self.relu(self.conv1(concatenated))
        depth1 = self.relu(self.cbam_dep_norm(self.cbam_dep(fused)))
        depth2 = self.relu(self.cbam_dep2_norm(self.cbam_dep2(fused)))
        
        batch_size, channels, height, width = depth2.size()

        theta_d=depth1.view(batch_size,channels,-1)
        theta_d=theta_d.permute(0,2,1)
        phi_d=depth2.view(batch_size,channels,-1)
        depth_mix=torch.matmul(theta_d,phi_d)
        depth_mix=F.softmax(depth_mix,dim=-1)

        g_s=sem.view(batch_size, self.dim,-1)
        g_s=g_s.permute(0,2,1)
        out = torch.matmul(depth_mix, g_s)
        out = out.permute(0,2,1).contiguous()
        out = out.view(batch_size, self.dim, height, width)
        out = self.shared_dep2_norm(self.shared_dep2(out)) #+ sem_embeds #* sem_embeds
        return out




