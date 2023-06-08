#!/usr/bin/python3.6

import torch
import numpy as np
import random


# This class generates batches of data for training and validation.
class BatchGenerator(object):
    # Initialize the object with input parameters and empty variables.
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list() # All examples of videos in a list.
        self.index = 0 # Pointer to the current example.
        self.num_classes = num_classes # Total number of classes.
        self.actions_dict = actions_dict # Dictionary containing mapping of action words to integer labels.
        self.gt_path = gt_path # Path to ground-truth files.
        self.features_path = features_path # Path to feature files.
        self.sample_rate = sample_rate # Step size for sampling the input.

    # Method to reset the pointer and shuffle the examples.
    def reset(self):
        self.index = 0 # Reset the pointer to the beginning.
        random.shuffle(self.list_of_examples) # Shuffle the list of examples randomly.

    # Method to check if there are more examples left.
    def has_next(self):
        if self.index < len(self.list_of_examples): # Check if the pointer is less than total number of examples.
            return True
        return False

    # Method to read the video names from the given file path and store them in a list.
    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1] # Store all the video names as examples in a list.
        print(self.list_of_examples)
        file_ptr.close()
        random.shuffle(self.list_of_examples) # Shuffle the list of examples randomly after reading them.

    # This function returns the input, output and mask tensors for the next batch of data based on the batch size.
    def next_batch(self, batch_size):
        print("batch_size", batch_size)
        # Take the subset of examples from self.list_of_examples based on batch_size.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size # Update the pointer after taking the subset.
        
        # Initialize the input and target lists for each example in the batch.
        batch_input = []
        batch_target = []
        
        # Loop through each example in the batch.
        for vid in batch:
            # Load the features and ground-truth file for the current example using numpy.
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            
            # Map the action words to integer labels using actions_dict for all the sentences in the ground-truth file.
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
                
            # Append the features and classes lists with self.sample_rate steps as inputs and outputs respectively.
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        # Compute length of sequences for all the targets and store in a variable.
        length_of_sequences = map(len, batch_target)
        
        # Initialize the input tensor with zeros using torch for all the examples in the batch.
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        
        # Initialize the target tensor with -100 using torch for all the examples in the batch.
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences, default=0), dtype=torch.long)*(-100)
        
        # Initialize the mask tensor with zeros using torch for all the examples in the batch.
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences, default=0), dtype=torch.float)
        
        # Loop through each example in the batch. 
        for i in range(len(batch_input)):
            # Fill the input tensor with the features and current length of sequence.
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            
            # Fill the target tensor with the classes and current length of sequence.
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            
            # Set the mask tensor to ones where we have ground-truth samples along the sequence.
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
