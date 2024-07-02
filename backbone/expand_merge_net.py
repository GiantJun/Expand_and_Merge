from typing import Callable, Iterable
from torch import nn
import torch
from backbone.clip import tokenize as clip_tokenizer
from transformers import BertTokenizer, BertModel
from backbone.clip_zoo import TextEncoder
from backbone.inc_net import load_clip_to_cpu, get_backbone

class ViT_Adapter_noAct(nn.Module):
    """
    x -> W_down -> GELU -> W_up -> 
    """
    def __init__(self, in_features:int, hidden_dim:int):
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        
        out = self.fc1(x)
        out = self.fc2(out)

        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()



class ExpandMergeNet(nn.Module):
    def __init__(self, logger, img_backbone_type, text_backbone_type, context:str, layer_names:Iterable[str], hidden_dim:int, mode:list):
        super().__init__()
        self._logger = logger
        self._context = context
        self._hidden_dim = hidden_dim
        self._mode = mode
        self._layer_names = layer_names
        self._forward_mode = 'mixture_adapter_mode'

        # img_backbone_type: clip_vit_b_p16, vit_base_patch16_224, .... (more could be found in backbone.vit.default_cfgs)
        # text_backbone_type: clip_text_b_p16, bert_base_cased
        self._img_backbone_type = img_backbone_type
        self._text_backbone_type = text_backbone_type

        self._need_img_proj = not ('clip' in img_backbone_type and 'clip' in text_backbone_type)
        if 'clip' in img_backbone_type or 'clip' in text_backbone_type:
            clip_model = load_clip_to_cpu('ViT-B/16')

        # text encoder
        if 'clip_text_b_p16' == text_backbone_type:
            self.token_embedding = clip_model.token_embedding
            self.text_encoder = TextEncoder(clip_model)
            self.text_embed_dim = self.text_encoder.embed_dim
            self.tokenizer = clip_tokenizer
        elif 'bert' in text_backbone_type:
            model_path = 'backbone/bert/{}'.format(text_backbone_type)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.text_encoder = BertModel.from_pretrained(model_path)
            self.text_embed_dim = 768
        else:
            raise ValueError('Unknown text backbone type: {} !'.format(text_backbone_type))
        
        # vision encoder
        if img_backbone_type == 'clip_vit_b_p16':
            self.image_encoder = clip_model.visual
            self.img_embed_dim = clip_model.visual.class_embedding.shape[0]
        else:
            self.image_encoder = get_backbone(logger=logger, backbone_type=img_backbone_type, pretrained=True)
            self.img_embed_dim = self.image_encoder.embed_dim
            self.image_encoder.head = nn.Identity()
        
        if self._need_img_proj:
            self.new_img_proj_alpha = nn.Parameter(torch.ones(1)*0.02)
            self.new_img_proj = nn.Linear(self.img_embed_dim, self.text_embed_dim)
            if 'wt_alpha_old' not in self._mode:
                self.old_img_proj_alpha = nn.Parameter(torch.ones(1)*0.02)
            self.old_img_proj = nn.Linear(self.img_embed_dim, self.text_embed_dim)

        self.known_class_names = []
        self.task_sizes = []
        self._is_skip_alpha = None

        self.output_features = {}

        self.set_hooks()
        
        model_dict = dict([*self.image_encoder.named_modules()])
        for layer_id in self._layer_names:
            adapter_id = layer_id.replace('.', '_')
            self.register_parameter(adapter_id+'_alpha_new', nn.Parameter(torch.ones(1)*0.02))
            self.register_module(adapter_id+'_adapter_new', ViT_Adapter_noAct(self.img_embed_dim, self._hidden_dim))
            
            if 'wt_alpha_old' not in self._mode:
                self.register_parameter(adapter_id+'_alpha_old', nn.Parameter(torch.ones(1)*0.02))
            self.register_module(adapter_id+'_adapter_old', ViT_Adapter_noAct(self.img_embed_dim, self._hidden_dim))
            
            layer = model_dict[layer_id]
            layer.register_forward_hook(self.apply_adapters(adapter_id))
    
    def forward_img_encoder(self, image):
        image_class_token = self.image_encoder(image)
        if self._need_img_proj and 'clip' in self._img_backbone_type:
            image_class_token = self.output_features['img_embeddings'][0]
        
        if self._need_img_proj:
            if self._forward_mode == 'mixture_adapter_mode':
                if len(self.task_sizes) == 1:
                    if self._is_skip_alpha:
                        image_class_token = self.new_img_proj(image_class_token)
                    else:
                        image_class_token = self.new_img_proj_alpha * self.new_img_proj(image_class_token)
                else:
                    new_image_class_token = self.new_img_proj(image_class_token)
                    old_image_class_token = self.old_img_proj(image_class_token)
                    if self._is_skip_alpha:
                        image_class_token = new_image_class_token + old_image_class_token
                    elif 'wt_alpha_old' not in self._mode:
                        image_class_token = self.new_img_proj_alpha * new_image_class_token + self.old_img_proj_alpha * old_image_class_token
                    else:
                        # 'wt_alpha_old' in self._mode:
                        image_class_token = self.new_img_proj_alpha * new_image_class_token + old_image_class_token

            elif self._forward_mode == 'old_adapter_mode':
                image_class_token = self.old_img_proj(image_class_token)
            elif self._forward_mode == 'new_adapter_mode':
                if self._is_skip_alpha:
                    image_class_token = self.new_img_proj(image_class_token)
                else:
                    image_class_token = self.new_img_proj_alpha * self.new_img_proj(image_class_token)
            else:
                raise ValueError('Unexpected forward mode {} in forward clip!'.format(self._mode))

        return image_class_token

    def forward_text_encoder(self):
        if 'clip' in self._text_backbone_type:
            text_class_token = self.text_encoder(self._tokenized_inputs, self._text_inputs)
        elif 'bert' in self._text_backbone_type:
            text_embeddings, text_class_token = self.text_encoder(input_ids=self._input_ids,
                                                                  attention_mask=self._attention_mask,
                                                                  token_type_ids=self._token_type_ids,
                                                                  return_dict=False)
            self.output_features['text_embeddings'] = text_embeddings
        else:
            raise ValueError('Unknown text backbone type: {} !'.format(self._text_backbone_type))
        
        return text_class_token

    def forward(self, image, T=1, skip_alpha=False):
        self._is_skip_alpha = skip_alpha
        image_class_tokens = self.forward_img_encoder(image)
        text_class_tokens = self.forward_text_encoder()
        
        image_feat = image_class_tokens / image_class_tokens.norm(dim=-1, keepdim=True)
        text_feat = text_class_tokens / text_class_tokens.norm(dim=-1, keepdim=True)

        logits = image_feat @ text_feat.t() / T

        self.output_features['features'] = image_class_tokens

        return logits, self.output_features
    
    def update_new_class_name(self, new_class_names):
        self.known_class_names.extend(new_class_names)
        self.task_sizes.append(len(new_class_names))

        class_names = [name.replace("_", " ") for name in self.known_class_names]
        prompts = [self._context.format(name) for name in class_names]
        if 'clip' in self._text_backbone_type:
            tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to(self.token_embedding.weight.device)
            with torch.no_grad():
                embedding = self.token_embedding(tokenized_prompts)
            if hasattr(self, '_tokenized_inputs'):
                self._tokenized_inputs = tokenized_prompts.to(self._tokenized_inputs.device)
                self._text_inputs = embedding.to(self._text_inputs.device)
            else:
                self.register_buffer('_tokenized_inputs', tokenized_prompts)
                self.register_buffer('_text_inputs', embedding)

        elif 'bert' in self._text_backbone_type:
            encoded_pair = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
            if hasattr(self, '_input_ids'):
                self._input_ids = encoded_pair['input_ids'].to(self._input_ids.device)
                self._attention_mask = encoded_pair['attention_mask'].to(self._attention_mask.device)
                self._token_type_ids = encoded_pair['token_type_ids'].to(self._token_type_ids.device)
            else:
                self.register_buffer('_input_ids', encoded_pair['input_ids'])
                self.register_buffer('_attention_mask', encoded_pair['attention_mask'])
                self.register_buffer('_token_type_ids', encoded_pair['token_type_ids'])
        else:
            raise ValueError('Unknown text backbone type: {} !'.format(self._text_backbone_type))
            
        if len(self.task_sizes) > 1:
            # merge old and new adapters
            for layer_id in self._layer_names:
                adapter_id = layer_id.replace('.', '_')
                
                if len(self.task_sizes) == 2:
                    new_state_dict = getattr(self, adapter_id+'_adapter_new').state_dict()
                    for key in new_state_dict.keys():
                            new_state_dict[key] = getattr(self, adapter_id+'_alpha_new').data * new_state_dict[key]
                    getattr(self, adapter_id+'_alpha_new').data = 0.02 * torch.ones(1, dtype=getattr(self, adapter_id+'_alpha_new').data.dtype)

                else:
                    old_state_dict = getattr(self, adapter_id+'_adapter_old').state_dict()
                    new_state_dict = getattr(self, adapter_id+'_adapter_new').state_dict()
                    if 'wt_alpha_old' not in self._mode:
                        for key in new_state_dict.keys():
                            new_state_dict[key] = (getattr(self, adapter_id+'_alpha_old').data * old_state_dict[key] + 
                                                getattr(self, adapter_id+'_alpha_new').data * new_state_dict[key])
                        getattr(self, adapter_id+'_alpha_old').data = 0.02 * torch.ones(1, dtype=getattr(self, adapter_id+'_alpha_old').data.dtype)
                        getattr(self, adapter_id+'_alpha_new').data = 0.02 * torch.ones(1, dtype=getattr(self, adapter_id+'_alpha_new').data.dtype)

                    else:
                        # 'wt_alpha_old' in self._mode:
                        for key in new_state_dict.keys():
                            new_state_dict[key] = (old_state_dict[key] + getattr(self, adapter_id+'_alpha_new').data * new_state_dict[key])
                        getattr(self, adapter_id+'_alpha_new').data = 0.02 * torch.ones(1, dtype=getattr(self, adapter_id+'_alpha_new').data.dtype)

                getattr(self, adapter_id+'_adapter_old').load_state_dict(new_state_dict)
                getattr(self, adapter_id+'_adapter_new').reset_parameters()

            if self._need_img_proj:
                if len(self.task_sizes) == 2:
                    new_state_dict = self.new_img_proj.state_dict()
                    for key in new_state_dict.keys():
                        new_state_dict[key] = self.new_img_proj_alpha.data * new_state_dict[key]
                    self.old_img_proj.load_state_dict(new_state_dict)
                    self.new_img_proj.reset_parameters()
                else:
                    old_state_dict = self.old_img_proj.state_dict()
                    new_state_dict = self.new_img_proj.state_dict()
                    if 'wt_alpha_old' not in self._mode:
                        for key in new_state_dict.keys():
                            new_state_dict[key] = (self.old_img_proj_alpha.data * old_state_dict[key] + 
                                                self.new_img_proj_alpha.data * new_state_dict[key])
                        self.old_img_proj_alpha.data = 0.02 * torch.ones(1, dtype=self.old_img_proj_alpha.dtype)
                        self.new_img_proj_alpha.data = 0.02 * torch.ones(1, dtype=self.new_img_proj_alpha.dtype)
                    else:
                        # 'wt_alpha_old' in self._mode:
                        for key in new_state_dict.keys():
                            new_state_dict[key] = old_state_dict[key] + self.new_img_proj_alpha.data * new_state_dict[key]
                        self.new_img_proj_alpha.data = 0.02 * torch.ones(1, dtype=self.new_img_proj_alpha.dtype)

                    self.old_img_proj.load_state_dict(new_state_dict)
                    self.new_img_proj.reset_parameters()
                
                
    def apply_adapters(self, adapter_id: str) -> Callable:
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            
            if self._forward_mode == 'old_adapter_mode':
                return getattr(self, adapter_id+'_adapter_old')(output)+output
            elif self._forward_mode == 'new_adapter_mode':
                if self._is_skip_alpha:
                    return getattr(self, adapter_id+'_adapter_new')(output) + output
                else:
                    return getattr(self, adapter_id+'_alpha_new') * getattr(self, adapter_id+'_adapter_new')(output) + output
            elif self._forward_mode == 'mixture_adapter_mode':
                if len(self.task_sizes) == 1:
                    if self._is_skip_alpha:
                        return getattr(self, adapter_id+'_adapter_new')(output) + output
                    else:
                        return getattr(self, adapter_id+'_alpha_new') * getattr(self, adapter_id+'_adapter_new')(output) + output
                else:
                    if self._is_skip_alpha:
                        return (getattr(self, adapter_id+'_adapter_new')(output) +
                                getattr(self, adapter_id+'_adapter_old')(output) +
                                output)
                    
                    elif 'wt_alpha_old' not in self._mode:
                        return (getattr(self, adapter_id+'_alpha_new') * getattr(self, adapter_id+'_adapter_new')(output) + 
                                getattr(self, adapter_id+'_alpha_old') * getattr(self, adapter_id+'_adapter_old')(output) + 
                                output)
                    else:
                        # 'wt_alpha_old' in self._mode:
                        return (getattr(self, adapter_id+'_alpha_new') * getattr(self, adapter_id+'_adapter_new')(output) + 
                                getattr(self, adapter_id+'_adapter_old')(output) + output)
            else:
                raise ValueError("Unknown network's forward mode!")
        return hook

    def set_hooks(self):
        if 'clip' in self._text_backbone_type:
            self.text_encoder.transformer.register_forward_hook(self.save_output_features('text_embeddings'))

        if 'clip' in self._img_backbone_type:
            self.image_encoder.transformer.register_forward_hook(self.save_output_features('img_embeddings'))
        else:
            # vit
            self.image_encoder.norm.register_forward_hook(self.save_output_features('img_embeddings'))
    
    def save_output_features(self, layer_id: str) -> Callable:
        def hook(module, input, output):
            self.output_features[layer_id] = output # LND
        return hook
    
    def mixture_adapter_mode(self):
        self._forward_mode = 'mixture_adapter_mode'

    def old_adapter_mode(self):
        self._forward_mode = 'old_adapter_mode'
    
    def new_adapter_mode(self):
        self._forward_mode = 'new_adapter_mode'

    def new_adapters_train(self):
        self.eval()
        for layer_id in self._layer_names:
            getattr(self, layer_id.replace('.', '_')+'_adapter_new').train()

    def freeze_new_adapters(self):
        for layer_id in self._layer_names:
            adapter_id = layer_id.replace('.', '_')
            for params in getattr(self, adapter_id+'_adapter_new').parameters():
                params.requires_grad = False
            getattr(self, adapter_id+'_adapter_new').eval()
    
    def activate_new_adapters(self):
        for layer_id in self._layer_names:
            adapter_id = layer_id.replace('.', '_')
            for params in getattr(self, adapter_id+'_adapter_new').parameters():
                params.requires_grad = True
            getattr(self, adapter_id+'_adapter_new').eval()

    def freeze_FE(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        self.image_encoder.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        if 'clip' in self._text_backbone_type:
            self.token_embedding.weight.requires_grad = False

        for layer_id in self._layer_names:
            adapter_id = layer_id.replace('.', '_')
            for params in getattr(self, adapter_id+'_adapter_old').parameters():
                params.requires_grad = False
            getattr(self, adapter_id+'_adapter_old').eval()
        
        if self._need_img_proj:
            for param in self.old_img_proj.parameters():
                param.requires_grad = False

        self._logger.info("Freezing vision encoder and old adapters ...")
