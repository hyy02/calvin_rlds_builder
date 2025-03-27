from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os


class Calvin(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(200, 200, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(84, 84, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float64,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'original_actions': 
                    tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            ' gripper velocities, 1x terminate episode].',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'idx_tuple':  tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        calvin_dataset_path = '/public/home/huangyiyang/data/task_D_D'
        return {
            'train': self._generate_examples(path=f'{calvin_dataset_path}/training'),
            'val': self._generate_examples(path=f'{calvin_dataset_path}/validation'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, idx_tuple, language_instruction):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            episode = []
            for step in range(idx_tuple[0], idx_tuple[1] + 1): # include the last frame
                step_data = np.load(os.path.join(episode_path, f'episode_{step:07d}.npz'), allow_pickle=True)
                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                episode.append({
                    'observation': {
                        'image': step_data['rgb_static'],
                        'wrist_image': step_data['rgb_gripper'],
                        'state': step_data['robot_obs'],
                    },
                    'action': step_data['rel_actions'],
                    'discount': 1.0,
                    'reward': 0,
                    'is_first': step == 0,
                    'is_last': step == (idx_tuple[1] - 1),
                    'is_terminal': i == (idx_tuple[1] - 1),
                    'language_instruction': language_instruction,
                    'original_actions': step_data['actions'],
                })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'idx_tuple': str(idx_tuple[0]) + ',' + str(idx_tuple[1]),
                    }
                }

                # if you want to skip an example for whatever reason, simply return None
                return str(idx_tuple[0]) + ',' + str(idx_tuple[1]), sample

        # The lang episodes for the data
        lang_ann = np.load(f'{path}/lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
        idx_tuple_list = lang_ann['info']['indx']
        ann_list = lang_ann['language']['ann']

        # for smallish datasets, use single-thread parsing
        for i, idx_tuple in enumerate(idx_tuple_list):
            yield _parse_example(path, idx_tuple, ann_list[i])

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

