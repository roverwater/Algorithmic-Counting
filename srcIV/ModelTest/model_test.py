import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import os
import math
from torch import nn, optim
from matplotlib.colors import ListedColormap
from srcIII import config
from srcIII.Model import counting_model_combined
from srcIII.Utils import RASPModel, utils 

# Check for GPU availability
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config.device}")

# Compile RASP and compute dataset
RASPModel.CompileRaspModel()
train_loader, val_loader, test_loader = utils.get_dataloaders('/home/ruov/projects/AlgorithmicCounting/n100_sq0p33_tr0p33_sz1-3_off-4-4_pat36',
                                                              batch_size=16,
                                                              split_ratios=(0.7, 0.15, 0.15),
                                                              seed=42)

# Move model to GPU
model = counting_model_combined.Model_Image(model_config=config.model_config,
                          model_parameters=config.model_parameters,
                          unembedding_data=config.unembed_matrix,
                          encoding_func=config.encoding_func,
                          encoded_vocab=config.encoded_vocab
                          ).to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

for batch_input, batch_labels in train_loader:
    # print(batch_input.shape)
    # print(batch_labels.shape)
    # output = model(batch_input)

    # print(output.argmax(-1))
    pass



def log_token_preferences(model, epoch, history):
    token_probs = F.softmax(model.classifier.token_logits, dim=0).detach().cpu().numpy()
    history[epoch] = token_probs

def plot_token_preferences(history):
    epochs = sorted(history.keys())

    probs_over_time = np.array([history[e] for e in epochs])
    # print(probs_over_time)
    
    plt.figure(figsize=(12, 8))
    vocab_size = probs_over_time.shape[1]
    
    for idx in range(vocab_size):
        token_name = idx
        plt.plot(epochs, probs_over_time[:, idx], label=f'Token {idx}: {token_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title(f'Evolution of Token Probabilities {config.vocab_list}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ImageLogits.png')

def get_current_temperature(epoch, total_epochs, initial_temp=1.0, final_temp=0.1, plateau_fraction=0):
    plateau_epochs = int(total_epochs * plateau_fraction)    
    if epoch < plateau_epochs:
        return initial_temp
    else:
        decay_epoch = epoch - plateau_epochs
        decay_total = total_epochs - plateau_epochs
        return initial_temp - (initial_temp - final_temp) * (decay_epoch / decay_total)
    
def convert_label_to_logits(batch_labels):
    batch_labels = batch_labels[:, 0].long()
    output_size = 36
    final_output_size = output_size + 3


    labels = torch.zeros(len(batch_labels), final_output_size, config.output_dim)

    one_hots = F.one_hot(batch_labels + 1, num_classes=config.output_dim).float()
    labels[:, 1:] = one_hots.unsqueeze(1).expand(-1, final_output_size - 1, -1)
    return labels

token_history = {}

def train_model():
    model.train()  
    for epoch in range(config.num_epochs):
        current_temp = get_current_temperature(
            epoch, 
            config.num_epochs, 
            initial_temp=config.start_temp, 
            final_temp=config.end_temp
        )
        total_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)  # Move data to GPU

            # print(labels[0])

            labels = convert_label_to_logits(labels)

            # print(inputs[:5].shape)
            # print(labels.shape)
            # print(labels[0].argmax(-1))


            if config.logit_dataset:
                labels = labels.argmax(dim=-1) 

                outputs = model(inputs, temperature=current_temp) 

                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                selected_outputs = outputs[:, 1:, :]  
                selected_labels = labels[:, 1:]       

                flattened_outputs = selected_outputs.contiguous().view(-1, outputs.size(-1)).to(config.device)
                flattened_labels = selected_labels.contiguous().view(-1).to(config.device)

                loss = criterion(flattened_outputs, flattened_labels)

                optimizer.zero_grad()
                torch.cuda.empty_cache()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)

            else:
                ###WORKS GOOD
                # Remove the line: labels = labels.argmax(dim=-1)

                outputs = model(inputs, temperature=current_temp)

                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                selected_outputs = outputs[:, 1:, :]  
                selected_labels = labels[:, 1:, :]    

                flattened_outputs = selected_outputs.contiguous().view(-1, selected_outputs.size(-1))
                flattened_labels = selected_labels.contiguous().view(-1, selected_labels.size(-1))

                log_probs = torch.nn.functional.log_softmax(flattened_outputs, dim=1)
                loss = torch.nn.functional.kl_div(log_probs, flattened_labels, reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                ###WORKS GOOD

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Temp: {current_temp:.3f}, Loss: {total_loss / len(train_loader):.4f}")
        evaluate_model()

def evaluate_model():
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)  # Move data to GPU
            labels = convert_label_to_logits(labels)
            labels = labels.argmax(dim=-1).to(config.device)  

            outputs = model(inputs, temperature=config.temperature)  
            predicted = outputs.argmax(dim=-1)

            # print(predicted.shape)
            # print(labels.shape)

            correct += (predicted[:, 1:] == labels[:, 1:]).sum().item()
            total += labels[:, 1:].numel()

    print(f"Test Accuracy: {correct / total:.4f}")

train_model()
plot_token_preferences(token_history)

# # #Visualize first batch
# # output_first_batch = model(test_input.to(config.device))
# # print(output_first_batch.shape)
# # outputs_after_training = output_first_batch.argmax(dim=-1)

# # cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# # num_samples_to_show = 25

# # columns = 7 
# # rows = math.ceil(num_samples_to_show / columns)
# # fig, axs = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
# # axs = axs.flatten()

# # for i in range(num_samples_to_show):
# #     original_img = test_input[i].squeeze().numpy()
# #     axs[i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
# #     axs[i].set_title(f"GT: {test_labels[i]}, Before: {outputs_before_training[i][1]}, After: {outputs_after_training[i][1] - 1}", fontsize=8)
# #     axs[i].axis('off')

# # for ax in axs[num_samples_to_show:]:
# #     ax.axis('off')

# # fig.suptitle("Model Predictions", fontsize=16, y=0.99)
# # plt.tight_layout()
# # plt.show()

# output_first_batch = model(test_input.to(config.device))
# outputs_after_training = output_first_batch.argmax(dim=-1)

# # cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# # num_samples_to_show = 25
# # columns = 7
# # rows = math.ceil(num_samples_to_show / columns)

# # fig, axs = plt.subplots(rows * 2, columns, figsize=(2 * columns, 4 * rows))  # doubled the rows to show side-by-side vertically
# # axs = axs.flatten()

# # # Pass test images through transformer
# # with torch.no_grad():
# #     conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config.device))
# #     pooled_out = model.image_transformer.pool_block(conv_out)

# # for i in range(num_samples_to_show):
# #     original_img = test_input[i].squeeze().cpu().numpy()
# #     transformed_img = pooled_out[i].squeeze().cpu().numpy()

# #     # Top row: Original
# #     axs[i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
# #     axs[i].set_title(f"GT: {test_labels[i]}\nBefore: {outputs_before_training[i][1]}, After: {outputs_after_training[i][1] - 1}", fontsize=7)
# #     axs[i].axis('off')

# #     # Bottom row: Transformed
# #     axs[i + num_samples_to_show].imshow(transformed_img, cmap='gray', interpolation='nearest')
# #     axs[i + num_samples_to_show].set_title("Transformed", fontsize=7)
# #     axs[i + num_samples_to_show].axis('off')

# # # Hide any extra axes
# # for ax in axs[2 * num_samples_to_show:]:
# #     ax.axis('off')

# # fig.suptitle("Original vs Transformed Images", fontsize=16, y=0.92)
# # plt.tight_layout()
# # plt.show()
# cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# num_samples_to_show = 25
# columns = 7
# rows = math.ceil(num_samples_to_show / columns)

# fig, axs = plt.subplots(rows, columns * 2, figsize=(2 * columns * 2, 4 * rows))  # doubled columns for side-by-side
# axs = axs.reshape(rows, columns * 2)

# # Get transformed output
# with torch.no_grad():
#     conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config.device))
#     pooled_out = model.image_transformer.pool_block(conv_out)

# for idx in range(num_samples_to_show):
#     row = idx // columns
#     col = (idx % columns) * 2  # Double column spacing for side-by-side

#     # Original image
#     original_img = test_input[idx].squeeze().cpu().numpy()
#     axs[row, col].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[row, col].set_title(
#         f"GT: {test_labels[idx]}\nBefore: {outputs_before_training[idx][1]}, After: {outputs_after_training[idx][1] - 1}", 
#         fontsize=7
#     )
#     axs[row, col].axis('off')

#     # Transformed image
#     transformed_img = pooled_out[idx].squeeze().cpu().numpy()
#     axs[row, col + 1].imshow(transformed_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[row, col + 1].set_title("Transformed", fontsize=7)
#     axs[row, col + 1].axis('off')

# # Hide any unused subplots
# total_axes = rows * columns * 2
# for ax in axs.flatten()[2 * num_samples_to_show:]:
#     ax.axis('off')

# fig.suptitle("Original and Transformed Images Side by Side", fontsize=16, y=0.95)
# plt.tight_layout()
# plt.show()

# os.system(r'find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf')


# # # 2. Test the data loaders
# # train_loader = config.train_loader
# # test_loader = config.test_loader

# # print(f"Training batches: {len(train_loader)}")
# # print(f"Test batches: {len(test_loader)}\n")

# # # 3. Inspect first batch
# # sample_batch = next(iter(train_loader))
# # inputs, labels = sample_batch
# # print(f"Batch shape - inputs: {inputs.shape}, labels: {labels.shape}")
# # print(f"Sample input shape: {inputs[0].shape}")
# # print(f"Sample label: {labels[0]}\n")

# # # 4. Fetch one batch (again, just to illustrate)
# # inputs, labels = next(iter(train_loader))

# # num_samples_to_show = 5
# # cmap = ListedColormap(['black', 'red', 'green', 'blue'])

# # # 5. Reference to the original ImageDataset (from the Subset in train_loader)
# # dataset_ref = train_loader.dataset.dataset  

# # # 6. Process and store the first five images in the batch
# # processed_images = []
# # for i in range(num_samples_to_show):
# #     np_img = inputs[i].squeeze().numpy()  # [32, 32]
# #     processed = dataset_ref._process_image(np_img)
# #     processed_images.append(processed)

# # # 7. Display original vs processed images
# # fig, axs = plt.subplots(2, num_samples_to_show, figsize=(3 * num_samples_to_show, 6))

# # for i in range(num_samples_to_show):
# #     # Top row: original image
# #     original_img = inputs[i].squeeze().numpy()
# #     axs[0, i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
# #     axs[0, i].set_title(f"Original: {labels[i].item()}")
# #     axs[0, i].axis('off')

# #     # Bottom row: processed image
# #     proc_img = processed_images[i].detach().numpy()
# #     axs[1, i].imshow(proc_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
# #     axs[1, i].set_title(f"Processed: {labels[i].item()}")
# #     axs[1, i].axis('off')

# # plt.tight_layout()
# # plt.savefig('dataset_test_1D_IMGAGE.png')

