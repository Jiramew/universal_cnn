from setuptools import setup

setup(
    name='ucnn',
    version='git.latest',
    description='Universal CNN Training & Predicting For Fixed-Length Label Captcha.',
    packages=["ucnn"],
    include_package_data=True,
    url='https://github.com/Jiramew/universal_cnn',
    license='BSD License',
    author='Jiramew',
    author_email='hanbingflying@sina.com',
    maintainer='Jiramew',
    maintainer_email='hanbingflying@sina.com',
    platforms=["all"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    scripts=['bin/ucnn'],
    install_requires=[
        'tensorflow>=1.4.0',
        'pillow>=4.2.1'
    ]
)
