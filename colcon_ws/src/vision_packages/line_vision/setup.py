from glob import glob

from setuptools import setup

package_name = 'line_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],   # training/ 은 넣지 않음
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ray',
    maintainer_email='rljw2066@gmail.com',
    description='경기장 경계선 인식 (UFLD-v2 + OpenVINO)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'line_vision_node = line_vision.line_vision_node:main',
        ],
    },
)
