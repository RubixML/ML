<a href="https://rubixml.com" target="_blank"><img src="https://raw.githubusercontent.com/RubixML/RubixML/master/docs/images/rubix-ml-logo.svg?sanitize=true" width="250" alt="Rubix ML for PHP" /></a>

[![PHP from Packagist](https://img.shields.io/packagist/php-v/rubix/ml.svg?style=flat-square&colorB=8892BF)](https://www.php.net/) [![Latest Stable Version](https://img.shields.io/packagist/v/rubix/ml.svg?style=flat-square&colorB=orange)](https://packagist.org/packages/rubix/ml) [![Downloads from Packagist](https://img.shields.io/packagist/dt/rubix/ml.svg?style=flat-square&colorB=red)](https://packagist.org/packages/rubix/ml) [![Travis](https://img.shields.io/travis/RubixML/RubixML.svg?style=flat-square)](https://travis-ci.org/RubixML/RubixML) [![GitHub license](https://img.shields.io/github/license/andrewdalpino/Rubix.svg?style=flat-square)](https://github.com/andrewdalpino/Rubix/blob/master/LICENSE.md)

A high-level machine learning and deep learning library for the [PHP](https://php.net) language.

- **Developer-friendly** API is straighforward and delightful
- **Modular** architecture combines power, flexibility, and extensibility
- **40+** modern supervised and unsupervised learning algorithms
- **Open source** and free to use commercially

## Installation
Install Rubix ML into your project with [Composer](https://getcomposer.org/):
```sh
$ composer require rubix/ml
```

## Requirements
- [PHP](https://php.net/manual/en/install.php) 7.1.3 or above

#### Optional

- [SVM extension](https://php.net/manual/en/book.svm.php) for Support Vector Machine engine (libsvm)
- [GD extension](https://php.net/manual/en/book.image.php) for image manipulation
- [Redis extension](https://github.com/phpredis/phpredis) for persisting to a Redis DB
- [Igbinary extension](https://github.com/igbinary/igbinary) for fast binary serialization of persistables

## Documentation


## Testing
Rubix utilizes a combination of static code analysis and unit tests. We provide three [Composer](https://getcomposer.org/) commands that can be run from the root directory to automate the testing process.

To run static analysis:
```sh
$ composer analyze
```

To run the coding style checker:
```sh
$ composer check
```

To run the unit tests:
```sh
$ composer test
```

## Contributing
See [CONTRIBUTING.md](https://github.com/RubixML/RubixML/blob/master/CONTRIBUTING.md) for guidelines.

## License
[MIT](https://github.com/RubixML/RubixML/blob/master/LICENSE.md)
