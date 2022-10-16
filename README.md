# MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video

Official implementation of CVPR 2022 paper([MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video](https://arxiv.org/abs/2203.00859)).

Note: Here are core codes of our work. We are organizing codes and prepare to submit to the [mmpose](https://github.com/open-mmlab/mmpose) as soon as possible.


<p align="center"> <img src="./assets/SittingDown_s1.gif" width="80%"> </p> 
<p align="center"> Visualization of our method and ground truth on Human3.6M </p>

## Environment

The code is conducted under the following environment:

* Ubuntu 18.04
* Python 3.6.10
* PyTorch 1.8.1
* CUDA 10.2

You can create the environment as follows:

```bash
conda env create -f requirements.yml
```

## Dataset

The Human3.6M dataset and HumanEva dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Please refer to it to set up the Human3.6M dataset (under ./data directory).

The MPI-INF-3DHP dataset setting follows the [MMPose](https://github.com/open-mmlab/mmpose).
Please refer it to set up the MPI-INF-3DHP dataset.


## Acknowledgement

Thanks for the baselines, we construct the code based on them:

* VideoPose3D
* SimpleBaseline


## Citation

```BibTeX
@InProceedings{Zhang_2022_CVPR,
    author    = {Zhang, Jinlu and Tu, Zhigang and Yang, Jianyu and Chen, Yujin and Yuan, Junsong},
    title     = {MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {13232-13242}
}
```

```

```
