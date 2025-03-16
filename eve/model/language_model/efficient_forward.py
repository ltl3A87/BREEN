# def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         images_highres: Optional[List[torch.FloatTensor]] = None,
#         image_sizes: Optional[List[List[int]]] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
# ) -> Union[Tuple, CausalLMOutputWithPast]:
#     # print("Before prepare_inputs_labels_for_multimodal...")
#     # print(input_ids.shape, labels.shape)
#     if inputs_embeds is None:
#         (
#             input_ids,
#             position_ids,
#             attention_mask,
#             past_key_values,
#             inputs_embeds,
#             labels
#         ) = self.prepare_inputs_labels_for_multimodal(
#             input_ids,
#             position_ids,
#             attention_mask,
#             past_key_values,
#             labels,
#             images,
#             images_highres,
#             image_sizes
#         )
#     # print("After prepare_inputs_labels_for_multimodal...")
#     # print(inputs_embeds.shape, labels.shape)
#
#     if labels is None:
#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )
#     else:
#         return self.forward_llm_efficient(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )
#
#
# def forward_llm_efficient(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels,
#                           use_cache, output_attentions, output_hidden_states, return_dict):
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#     # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#     outputs = self.model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         position_ids=position_ids,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )
#
#     hidden_states = outputs[0]
#     hidden_dim = hidden_states.size(-1)
#     shift_labels = labels[..., 1:].contiguous().reshape(-1)
#     shift_hidden_states = hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
#     assert shift_labels.size(0) == shift_hidden_states.size(0)
#     mask = shift_labels > -1
#
#     seen_tokens = mask.float().sum().item()
#     if not seen_tokens > 0:
#         logits = self.lm_head(shift_hidden_states[0:2])
#         loss = logits.sum() * 0
#     else:
#         shift_labels = shift_labels[mask]
#         shift_hidden_states = shift_hidden_states[mask, :]
#         logits = self.lm_head(shift_hidden_states)
#         logits = logits.float()
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(logits, shift_labels)
#
#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output
#
#     return CausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#     )
