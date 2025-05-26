# import os
# import scipy.io
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# # Create a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# plt.rcParams.update({
#     'font.size': 20,
#     'axes.titlesize': 100,
#     'axes.labelsize': 100,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
# })

# # Common function for ShanghaiTech parts
# def plot_shanghai_distribution(ax, root_path, part_name, bin_width):
#     train_counts = []
#     test_counts = []

#     for data_type in ['train_data', 'test_data']:
#         data_dir = os.path.join(root_path, data_type)
#         gt_dir = os.path.join(data_dir, 'ground_truth')
        
#         for mat_file in os.listdir(gt_dir):
#             if mat_file.endswith('.mat'):
#                 mat_path = os.path.join(gt_dir, mat_file)
#                 mat_data = scipy.io.loadmat(mat_path)
#                 count = mat_data['image_info'][0][0]['number'][0][0][0][0]
                
#                 if data_type == 'train_data':
#                     train_counts.append(count)
#                 else:
#                     test_counts.append(count)

#     # Calculate bins and ticks
#     min_val = min(min(train_counts), min(test_counts))
#     max_val = max(max(train_counts), max(test_counts))
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     xtick_values = np.arange(
#         (min_val // bin_width) * bin_width,
#         max_val + bin_width, 
#         bin_width
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     # Plotting
#     ax.hist([train_counts, test_counts], 
#             bins=bins,
#             label=['Train', 'Test'],
#             edgecolor='black',
#             alpha=0.7,
#             color=['#1f77b4', '#ff7f0e'])
    
#     # Axis configuration
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)', fontsize=12)
#     ax.set_ylabel('Number of Images', fontsize=12)
#     ax.set_title(f'Crowd Count Distribution: {part_name}', fontsize=14)
#     ax.legend(title='Dataset Split')
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

#     # # Add count labels
#     # for rect in ax.patches:
#     #     height = rect.get_height()
#     #     if height > 0:
#     #         ax.text(rect.get_x() + rect.get_width()/2, height + 0.5,
#     #                 f'{int(height)}', ha='center', va='bottom', fontsize=8)

# # Function for UCF_CC_50 dataset
# def plot_ucf_distribution(ax, root_path, part_name, bin_width):
#     counts = []
#     for img_file in os.listdir(root_path):
#         if img_file.endswith('.jpg'):
#             mat_file = os.path.join(root_path, f'{os.path.splitext(img_file)[0]}_ann.mat')
#             mat_data = scipy.io.loadmat(mat_file)
#             counts.append(mat_data['annPoints'].shape[0])

#     # Calculate bins and ticks
#     min_val = min(counts)
#     max_val = max(counts)
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     xtick_values = np.arange(
#         (min_val // bin_width) * bin_width,
#         max_val + bin_width, 
#         bin_width
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     # Plotting
#     ax.hist(counts, 
#             bins=bins,
#             edgecolor='black',
#             alpha=0.7,
#             color='#2ca02c')
    
#     # Axis configuration
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)', fontsize=12)
#     ax.set_ylabel('Number of Images', fontsize=12)
#     ax.set_title(f'Crowd Count Distribution: {part_name}', fontsize=14)
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

#     # # Add count labels
#     # for rect in ax.patches:
#     #     height = rect.get_height()
#     #     if height > 0:
#     #         ax.text(rect.get_x() + rect.get_width()/2, height + 0.5,
#     #                 f'{int(height)}', ha='center', va='bottom', fontsize=8)

# # Function for FSC147 dataset
# def plot_fsc_distribution(ax, counts, part_name, bin_width):
#     # Calculate bins and ticks
#     min_val = min(counts)
#     max_val = max(counts)
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     xtick_values = np.arange(
#         (min_val // bin_width) * bin_width,
#         max_val + bin_width, 
#         bin_width
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     # Plotting
#     ax.hist(counts, 
#             bins=bins,
#             edgecolor='black',
#             alpha=0.7,
#             color='#9467bd')
    
#     # Axis configuration
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)', fontsize=12)
#     ax.set_ylabel('Number of Images', fontsize=12)
#     ax.set_title(f'Crowd Count Distribution: {part_name}', fontsize=14)
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

#     # # Add count labels
#     # for rect in ax.patches:
#     #     height = rect.get_height()
#     #     if height > 0:
#     #         ax.text(rect.get_x() + rect.get_width()/2, height + 0.5,
#     #                 f'{int(height)}', ha='center', va='bottom', fontsize=8)

# # Process FSC147 dataset
# dataset_path = "/media/ruov/ae95a350-5b34-4fde-bc56-baef9790273f1/home/ruov/Documents/FSC147_384_V2"  # Update this path
# images_dir = os.path.join(dataset_path, "images_384_VarV2")
# density_dir = os.path.join(dataset_path, "gt_density_map_adaptive_384_VarV2")

# image_counts = []
# for img_file in os.listdir(images_dir):
#     if img_file.endswith('.jpg'):
#         base_name = os.path.splitext(img_file)[0]
#         density_path = os.path.join(density_dir, f"{base_name}.npy")
#         density_map = np.load(density_path)
#         count = np.sum(density_map).round().astype(int)
#         image_counts.append(count)

# # Plot ShanghaiTech Part A
# plot_shanghai_distribution(
#     axs[0,0],
#     root_path='ShanghaiTech_Crowd_Counting_Dataset/part_A_final',
#     part_name='Part A',
#     bin_width=100
# )

# # Plot ShanghaiTech Part B
# plot_shanghai_distribution(
#     axs[0,1],
#     root_path='ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
#     part_name='Part B',
#     bin_width=20
# )

# # Plot UCF_CC_50
# plot_ucf_distribution(
#     axs[1,0],
#     root_path='UCF_CC_50',
#     part_name='UCF_CC_50',
#     bin_width=100
# )

# # Plot FSC147
# plot_fsc_distribution(
#     axs[1,1],
#     counts=image_counts,
#     part_name='FSC147',
#     bin_width=50  # Adjust based on your data distribution
# )

# plt.tight_layout()
# plt.show()

# import os
# import scipy.io
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# # Create a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# # Adjusted plotting parameters for better readability
# plt.rcParams.update({
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
# })

# # Common function for ShanghaiTech parts
# def plot_shanghai_distribution(ax, root_path, part_name, bin_width):
#     # [Existing implementation unchanged]
#     train_counts = []
#     test_counts = []

#     for data_type in ['train_data', 'test_data']:
#         data_dir = os.path.join(root_path, data_type)
#         gt_dir = os.path.join(data_dir, 'ground_truth')
        
#         for mat_file in os.listdir(gt_dir):
#             if mat_file.endswith('.mat'):
#                 mat_path = os.path.join(gt_dir, mat_file)
#                 mat_data = scipy.io.loadmat(mat_path)
#                 count = mat_data['image_info'][0][0]['number'][0][0][0][0]
                
#                 if data_type == 'train_data':
#                     train_counts.append(count)
#                 else:
#                     test_counts.append(count)

#     # Calculate bins and ticks
#     min_val = min(min(train_counts), min(test_counts))
#     max_val = max(max(train_counts), max(test_counts))
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     xtick_values = np.arange(
#         (min_val // bin_width) * bin_width,
#         max_val + bin_width, 
#         bin_width
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     # Plotting
#     ax.hist([train_counts, test_counts], 
#             bins=bins,
#             label=['Train', 'Test'],
#             edgecolor='black',
#             alpha=0.7,
#             color=['#1f77b4', '#ff7f0e'])
    
#     # Axis configuration
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)')
#     ax.set_ylabel('Number of Images')
#     ax.set_title(f'Crowd Count Distribution: {part_name}')
#     ax.legend(title='Dataset Split')
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

# # Function for UCF_CC_50 dataset (unchanged)
# def plot_ucf_distribution(ax, root_path, part_name, bin_width):
#     # [Existing implementation unchanged]
#     counts = []
#     for img_file in os.listdir(root_path):
#         if img_file.endswith('.jpg'):
#             mat_file = os.path.join(root_path, f'{os.path.splitext(img_file)[0]}_ann.mat')
#             mat_data = scipy.io.loadmat(mat_file)
#             counts.append(mat_data['annPoints'].shape[0])

#     min_val = min(counts)
#     max_val = max(counts)
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     xtick_values = np.arange(
#         (min_val // bin_width) * bin_width,
#         max_val + bin_width, 
#         bin_width
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     ax.hist(counts, 
#             bins=bins,
#             edgecolor='black',
#             alpha=0.7,
#             color='#2ca02c')
    
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)')
#     ax.set_ylabel('Number of Images')
#     ax.set_title(f'Crowd Count Distribution: {part_name}')
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

# # Modified FSC147 plotting function
# def plot_fsc_distribution(ax, counts, part_name, bin_width, tick_step=None):
#     min_val = min(counts)
#     max_val = max(counts)
    
#     # Allow custom tick interval (default to bin_width)
#     if tick_step is None:
#         tick_step = bin_width
        
#     bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
#     # Generate x-ticks with specified interval
#     xtick_values = np.arange(
#         (min_val // tick_step) * tick_step,
#         max_val + tick_step, 
#         tick_step
#     )
#     xtick_labels = [f"{int(x)}" for x in xtick_values]

#     # Plotting
#     ax.hist(counts, 
#             bins=bins,
#             edgecolor='black',
#             alpha=0.7,
#             color='#9467bd')
    
#     # Set log scale for y-axis
#     ax.set_yscale('log')
    
#     # Axis configuration
#     ax.set_xticks(xtick_values + bin_width/2)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
#     ax.set_xlabel('Object Count (Binned)')
#     ax.set_ylabel('Number of Images (log scale)')
#     ax.set_title(f'Crowd Count Distribution: {part_name}')
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

# # Process FSC147 dataset
# dataset_path = "/media/ruov/ae95a350-5b34-4fde-bc56-baef9790273f1/home/ruov/Documents/FSC147_384_V2"
# images_dir = os.path.join(dataset_path, "images_384_VarV2")
# density_dir = os.path.join(dataset_path, "gt_density_map_adaptive_384_VarV2")

# image_counts = []
# for img_file in os.listdir(images_dir):
#     if img_file.endswith('.jpg'):
#         base_name = os.path.splitext(img_file)[0]
#         density_path = os.path.join(density_dir, f"{base_name}.npy")
#         density_map = np.load(density_path)
#         count = np.sum(density_map).round().astype(int)
#         image_counts.append(count)

# # Plot ShanghaiTech Part A
# plot_shanghai_distribution(
#     axs[0,0],
#     root_path='ShanghaiTech_Crowd_Counting_Dataset/part_A_final',
#     part_name='Part A',
#     bin_width=100
# )

# # Plot ShanghaiTech Part B
# plot_shanghai_distribution(
#     axs[0,1],
#     root_path='ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
#     part_name='Part B',
#     bin_width=20
# )

# # Plot UCF_CC_50
# plot_ucf_distribution(
#     axs[1,0],
#     root_path='UCF_CC_50',
#     part_name='UCF_CC_50',
#     bin_width=100
# )

# # Plot FSC147 with modified parameters
# plot_fsc_distribution(
#     axs[1,1],
#     counts=image_counts,
#     part_name='FSC147',
#     bin_width=50,
#     tick_step=100  # Reduced x-axis tick density
# )

# plt.tight_layout()
# plt.show()

import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Global styling parameters
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16
})

# ShanghaiTech Plotting Function
def plot_shanghai_distribution(ax, root_path, part_name, bin_width):
    train_counts = []
    test_counts = []

    for data_type in ['train_data', 'test_data']:
        data_dir = os.path.join(root_path, data_type)
        gt_dir = os.path.join(data_dir, 'ground_truth')
        
        for mat_file in os.listdir(gt_dir):
            if mat_file.endswith('.mat'):
                mat_path = os.path.join(gt_dir, mat_file)
                mat_data = scipy.io.loadmat(mat_path)
                count = mat_data['image_info'][0][0]['number'][0][0][0][0]
                
                if data_type == 'train_data':
                    train_counts.append(count)
                else:
                    test_counts.append(count)

    # Linear bin calculation
    min_val = min(min(train_counts), min(test_counts))
    max_val = max(max(train_counts), max(test_counts))
    bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
    # Plotting
    ax.hist([train_counts, test_counts], 
            bins=bins,
            label=['Train', 'Test'],
            edgecolor='black',
            alpha=0.7,
            color=['#1f77b4', '#ff7f0e'])
    
    # Axis configuration
    ax.set_xlabel('Object Count (Binned)', fontsize=20)
    ax.set_ylabel('Number of Images', fontsize=20)
    ax.set_title(f'ShanghaiTech: {part_name}', fontsize=22)
    ax.legend(title='Split')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

# UCF_CC_50 Plotting Function
def plot_ucf_distribution(ax, root_path, part_name, bin_width):
    counts = []
    for img_file in os.listdir(root_path):
        if img_file.endswith('.jpg'):
            mat_file = os.path.join(root_path, f'{os.path.splitext(img_file)[0]}_ann.mat')
            mat_data = scipy.io.loadmat(mat_file)
            counts.append(mat_data['annPoints'].shape[0])

    # Linear bin calculation
    min_val = min(counts)
    max_val = max(counts)
    bins = np.arange(min_val - 0.5, max_val + bin_width, bin_width)
    
    ax.hist(counts, 
            bins=bins,
            edgecolor='black',
            alpha=0.7,
            color='#2ca02c')
    
    # Axis configuration
    ax.set_xlabel('Object Count (Binned)', fontsize=20)
    ax.set_ylabel('Number of Images', fontsize=20)
    ax.set_title(f'UCF_CC_50 Distribution', fontsize=22)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

# FSC147 Plotting Function with Log X-axis
def plot_fsc_distribution(ax, counts, part_name):
    min_val = max(1, min(counts))  # Ensure minimum > 0 for log scale
    max_val = max(counts)
    
    # Logarithmic bin calculation
    bins = np.logspace(np.log10(min_val), 
                      np.log10(max_val), 
                      num=15,
                      base=10)

    ax.hist(counts, 
            bins=bins,
            edgecolor='black',
            alpha=0.7,
            color='#9467bd')
    
    # Log scale configuration
    ax.set_xscale('log')
    ax.set_xlabel('Object Count (Log Scale)', fontsize=20)
    ax.set_ylabel('Number of Images', fontsize=20)
    ax.set_title(f'FSC147 Distribution', fontsize=22)
    
    # Tick formatting
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Process FSC147 dataset
dataset_path = "/media/ruov/ae95a350-5b34-4fde-bc56-baef9790273f1/home/ruov/Documents/FSC147_384_V2"
images_dir = os.path.join(dataset_path, "images_384_VarV2")
density_dir = os.path.join(dataset_path, "gt_density_map_adaptive_384_VarV2")

image_counts = []
for img_file in os.listdir(images_dir):
    if img_file.endswith('.jpg'):
        base_name = os.path.splitext(img_file)[0]
        density_path = os.path.join(density_dir, f"{base_name}.npy")
        density_map = np.load(density_path)
        count = np.sum(density_map).round().astype(int)
        image_counts.append(count)

# Plot all datasets
plot_shanghai_distribution(axs[0,0], 
                          'ShanghaiTech_Crowd_Counting_Dataset/part_A_final',
                          'Part A', 
                          bin_width=100)

plot_shanghai_distribution(axs[0,1], 
                          'ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
                          'Part B', 
                          bin_width=20)

plot_ucf_distribution(axs[1,0], 
                     'UCF_CC_50',
                     'UCF_CC_50', 
                     bin_width=100)

plot_fsc_distribution(axs[1,1], 
                     image_counts, 
                     'FSC147')

plt.tight_layout(pad=3.0)
plt.show()

# import os
# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# # 1) Use a clean, professional style
# plt.style.use("seaborn-v0_8-whitegrid")

# # 2) Choose a colorblind-friendly qualitative palette
# cmap = plt.get_cmap('tab10')
# COLORS = {
#     'train': cmap(0),
#     'test':  cmap(1),
#     'ucf':   cmap(2),
#     'fsc':   cmap(3),
# }

# # Create a 2×2 grid with constrained layout
# fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)

# def format_axes(ax, title):
#     """Common formatting for all axes."""
#     # Center the title, bump it up a bit
#     ax.set_title(title, fontsize=16, loc='center', pad=15)
#     # X / Y labels
#     ax.set_xlabel('Object count (binned)', fontsize=14)
#     ax.set_ylabel('Number of Images', fontsize=14)
#     # Auto-locator to reduce tick crowding
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
#     ax.tick_params(axis='x', rotation=45, labelsize=12)
#     ax.tick_params(axis='y', labelsize=12)
#     # Legend inside, no frame
#     ax.legend(loc='upper right', frameon=False, fontsize=12, title='Split')
#     # Light y-grid only
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.set_axisbelow(True)

# def annotate_bars(ax):
#     # """Add counts above each bar."""
#     # for rect in ax.patches:
#     #     h = rect.get_height()
#     #     if h > 0:
#     #         ax.text(
#     #             rect.get_x() + rect.get_width() / 2, 
#     #             h + 0.5, 
#     #             f'{int(h)}', 
#     #             ha='center', va='bottom', 
#     #             fontsize=10
#     #         )
#     pass

# def plot_shanghai(ax, path, name, bw):
#     train, test = [], []
#     for split in ['train_data','test_data']:
#         gt = os.path.join(path, split, 'ground_truth')
#         for m in os.listdir(gt):
#             if not m.endswith('.mat'): continue
#             data = scipy.io.loadmat(os.path.join(gt,m))
#             cnt  = int(data['image_info'][0][0]['number'][0][0][0][0])
#             (train if split=='train_data' else test).append(cnt)

#     mn, mx = min(train+test), max(train+test)
#     bins = np.arange(mn - .5, mx + bw, bw)
#     ax.hist(
#         [train, test], bins=bins, 
#         color=[COLORS['train'], COLORS['test']],
#         label=['Train','Test'],
#         edgecolor='white', alpha=0.8
#     )
#     format_axes(ax, f'ShanghaiTech {name}')
#     annotate_bars(ax)

# def plot_ucf(ax, path, bw):
#     counts = []
#     for img in os.listdir(path):
#         if not img.endswith('.jpg'): continue
#         ann = os.path.splitext(img)[0] + '_ann.mat'
#         data = scipy.io.loadmat(os.path.join(path, ann))
#         counts.append(data['annPoints'].shape[0])

#     mn, mx = min(counts), max(counts)
#     bins = np.arange(mn - .5, mx + bw, bw)
#     ax.hist(
#         counts, bins=bins,
#         color=COLORS['ucf'],
#         edgecolor='white', alpha=0.8,
#         label='Images'
#     )
#     format_axes(ax, 'UCF_CC_50')
#     annotate_bars(ax)

# def plot_fsc(ax, counts, bw):
#     mn, mx = min(counts), max(counts)
#     bins = np.arange(mn - .5, mx + bw, bw)
#     ax.hist(
#         counts, bins=bins,
#         color=COLORS['fsc'],
#         edgecolor='white', alpha=0.8,
#         label='Images'
#     )
#     format_axes(ax, 'FSC147')
#     annotate_bars(ax)

# def plot_fsc_log(ax, counts, num_bins=50):
#     # Compute log-spaced bins between the min (≥1) and max count
#     mn, mx = max(min(counts), 1), max(counts)
#     bins = np.logspace(np.log10(mn), np.log10(mx), num=num_bins)

#     # Plot
#     ax.hist(
#         counts,
#         bins=bins,
#         color=COLORS['fsc'],
#         edgecolor='white',
#         alpha=0.8,
#         label='Images'
#     )

#     # Switch to log scale on x-axis
#     ax.set_xscale('log')
#     # If you also want the counts axis (y) in log:
#     # ax.set_yscale('log')

#     # Tidy up ticks: show a few “round” ticks
#     from matplotlib.ticker import LogLocator, FormatStrFormatter
#     ax.xaxis.set_major_locator(LogLocator(base=10))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#     ax.tick_params(axis='x', rotation=45, labelsize=12)

#     # Apply your usual formatting
#     format_axes(ax, 'FSC147 (log-scale)')
#     # (no need to annotate bars when log scale—they often overlap)

# def plot_fsc_clipped(ax, counts, bin_width, max_count=800):
#     # 1) Keep only counts ≤ max_count so no bars beyond that
#     clipped = [c for c in counts if c <= max_count]

#     # 2) Build linear bins from 0→max_count
#     bins = np.arange(0 - 0.5, max_count + bin_width, bin_width)

#     # 3) Plot exactly as before, but on a linear axis
#     ax.hist(
#         clipped,
#         bins=bins,
#         color=COLORS['fsc'],
#         edgecolor='white',
#         alpha=0.8,
#         label=f'Images (≤{max_count})'
#     )

#     # 4) Force x-axis limits to [0, max_count]
#     ax.set_xlim(0, max_count)

#     # 5) Re-apply your shared formatting
#     format_axes(ax, f'FSC147 (clipped ≤ {max_count})')
#     # no log scale, so nothing else to remove



# # ——— Load your FSC147 counts as before ———
# dataset_path  = "/media/ruov/ae95a350-5b34-4fde-bc56-baef9790273f1/home/ruov/Documents/FSC147_384_V2"
# images_dir    = os.path.join(dataset_path, "images_384_VarV2")
# density_dir   = os.path.join(dataset_path, "gt_density_map_adaptive_384_VarV2")

# fsc_counts = []
# for jpg in os.listdir(images_dir):
#     if not jpg.endswith('.jpg'): continue
#     base = os.path.splitext(jpg)[0]
#     dm   = np.load(os.path.join(density_dir, f"{base}.npy"))
#     fsc_counts.append(int(np.sum(dm).round()))

# # ——— Plot each subplot ———
# plot_shanghai(
#     axs[0,0],
#     path='ShanghaiTech_Crowd_Counting_Dataset/part_A_final',
#     name='Part A',
#     bw=100
# )
# plot_shanghai(
#     axs[0,1],
#     path='ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
#     name='Part B',
#     bw=20
# )
# plot_ucf(
#     axs[1,0],
#     path='UCF_CC_50',
#     bw=100
# )
# # plot_fsc(
# #     axs[1,1],
# #     counts=fsc_counts,
# #     bw=50
# # )
# # plot_fsc_log(
# #     axs[1,1], 
# #     fsc_counts, 
# #     num_bins=40
# # )
# plot_fsc_clipped(
#     axs[1,1],
#     counts=fsc_counts,
#     bin_width=5,
#     max_count=400
# )

# # Overall figure title
# fig.suptitle('Crowd Count Distributions Across Four Benchmarks', fontsize=18, y=1.02)

# plt.show()


