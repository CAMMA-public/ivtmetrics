from distutils.core import setup

setup(
    name='ivtmetrics',
    version='0.0.1',    
    packages=['ivtmetrics'],
    author='Chinedu Nwoye',
    author_email='nwoye@unistra.fr',    
    description='A Python evaluation metrics package for action triplet recognition',
    keywords = ['triplet', 'average precision', 'AP'], 
    url='https://github.com/CAMMA-public/ivtmetrics',
    download_url = 'https://github.com/CAMMA-public/ivtmetrics/archive/refs/tags/v0.0.1.tar.gz',    # I explain this later on
    license='BSD 2-clause', # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    install_requires=['scikit-learn',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research', # Define that your audience are developers
        'License :: OSI Approved :: BSD License',  
        'Topic :: Software Development :: Build Tools',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',   
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)    

# guide @ https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
# 'Operating System :: POSIX :: Linux :: Windows :: Mac :: Unix',   