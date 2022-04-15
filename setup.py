from setuptools import setup, find_packages

setup(
	name='project2',
	version='1.0',
	author='Sai Sharath Kumar, Sabbarapu',
	authour_email='s.sharat.kumar@ou.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']
)