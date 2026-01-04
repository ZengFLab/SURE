from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SURE-tools',
    version='3.2.31',
    description='Succinct Representation of Single Cells',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Feng Zeng',
    author_email='zengfeng@xmu.edu.cn',
    packages=find_packages(),
    install_requires=['dill==0.3.8','scanpy','pytorch-ignite','datatable','scipy','numpy','scikit-learn','pandas','pyro-ppl', "jax[cuda12]",
                      'leidenalg','python-igraph','networkx','matplotlib','seaborn','fa2-modified','zuko'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    url='https://github.com/ZengFLab/SURE',  # 项目的 GitHub 地址

    entry_points={
        'console_scripts': [
            'SURE=SURE.SURE:main',  # 允许用户通过命令行调用 main 函数
            'PerturbFlow=PerturbFlow.PerturbFlow:main'
        ],
    },
)