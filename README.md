
# Rubix ML

[![PHP from Packagist](https://img.shields.io/packagist/php-v/rubix/ml.svg?style=flat&colorB=8892BF)](https://www.php.net/) [![Latest Stable Version](https://img.shields.io/packagist/v/rubix/ml.svg?style=flat&colorB=orange)](https://packagist.org/packages/rubix/ml) [![Downloads from Packagist](https://img.shields.io/packagist/dt/rubix/ml.svg?style=flat&colorB=red)](https://packagist.org/packages/rubix/ml) [![Code Checks](https://github.com/RubixML/ML/actions/workflows/ci.yml/badge.svg)](https://github.com/RubixML/ML/actions/workflows/ci.yml) [![GitHub](https://img.shields.io/github/license/RubixML/RubixML)](https://github.com/RubixML/ML/blob/master/LICENSE.md)

A modern, high-level machine learning and deep learning library built for the [PHP 8+](https://php.net) ecosystem.

- **Developer-first** API designed for clarity and ease of use
- **40+** cutting-edge supervised and unsupervised learning algorithms
- Comprehensive support for ETL, feature preprocessing, and cross-validation workflows
- Fully **open source** with commercial-friendly licensing

## Installation
Add Rubix ML to your project via [Composer](https://getcomposer.org/):
```bash
composer require rubix/ml
````

### Requirements

* [PHP](https://www.php.net/manual/en/install.php) version **8.0 or higher**

#### Highly Recommended

* [Tensor extension](https://github.com/RubixML/Tensor) for accelerated matrix and vector computations leveraging PHP 8's JIT capabilities

#### Optional Extensions

* [GD extension](https://www.php.net/manual/en/book.image.php) for advanced image processing features
* [Mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for efficient multibyte string handling
* [SVM extension](https://www.php.net/manual/en/book.svm.php) to leverage libsvm as an alternative Support Vector Machine backend
* [PDO extension](https://www.php.net/manual/en/book.pdo.php) for seamless relational database integration
* [GraphViz](https://graphviz.org/) to generate and visualize model graphs and pipelines

## Documentation

Explore the latest documentation and guides [here](https://docs.rubixml.com).

## What is Rubix ML?

Rubix ML is an open-source machine learning library written entirely in PHP 8+, empowering you to build intelligent applications that learn from data. It provides a comprehensive suite of tools to support the entire ML lifecycle: from data ingestion and transformation (ETL), through model training and validation, to deployment in production. With a robust collection of over 40 supervised and unsupervised algorithms, Rubix ML is designed to fit naturally into your PHP projects, harnessing modern language features for performance and developer productivity.

## Getting Started

If you’re new to machine learning, start with the [What is Machine Learning?](https://docs.rubixml.com/latest/what-is-machine-learning.html) section to get a solid conceptual foundation. For those familiar with ML concepts, the [basic introduction](https://docs.rubixml.com/latest/basic-introduction.html) offers a concise overview of a typical Rubix ML workflow. Then dive into the official tutorials below, tailored for all skill levels from beginner to advanced.

### Tutorials & Example Projects

Explore real-world projects built with Rubix ML — all include setup instructions and preprocessed datasets to get you coding quickly:

* [CIFAR-10 Image Recognizer](https://github.com/RubixML/CIFAR-10)
* [Color Clusterer](https://github.com/RubixML/Colors)
* [Credit Default Risk Predictor](https://github.com/RubixML/Credit)
* [Customer Churn Predictor](https://github.com/RubixML/Churn)
* [Divorce Predictor](https://github.com/RubixML/Divorce)
* [DNA Taxonomer](https://github.com/RubixML/DNA)
* [Dota 2 Game Outcome Predictor](https://github.com/RubixML/Dota2)
* [Human Activity Recognizer](https://github.com/RubixML/HAR)
* [Housing Price Predictor](https://github.com/RubixML/Housing)
* [Iris Flower Classifier](https://github.com/RubixML/Iris)
* [MNIST Handwritten Digit Recognizer](https://github.com/RubixML/MNIST)
* [Text Sentiment Analyzer](https://github.com/RubixML/Sentiment)
* [Titanic Survival Predictor](https://github.com/Jenutka/titanic_php)

## Community

Connect, collaborate, and get support from other Rubix ML users:

* [Join Our Telegram Channel](https://t.me/RubixML)

## Contributing

Interested in contributing? See the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and best practices.

## License

Rubix ML is licensed under the [MIT License](LICENSE) and the documentation is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
