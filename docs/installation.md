# Installation

Install Rubix ML into your project using [Composer](https://getcomposer.org/):

```sh
composer require rubix/ml
```

Optionally, install the Tensor extension using [PIE](https://www.php.net/manual/en/install.pie.intro.php) like in the example below:

```sh
pie install rubix/tensor_ext:^3.0
```

## Requirements

- [PHP](https://php.net/manual/en/install.php) 7.4 or above

### Recommended

- [Tensor 3.x extension](https://github.com/RubixML/Tensor-Ext) for fast Matrix/Vector computing

### Optional

- [GD extension](https://php.net/manual/en/book.image.php) for image support
- [Mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for fast multibyte string manipulation
- [SVM extension](https://php.net/manual/en/book.svm.php) for Support Vector Machine engine (libsvm)
- [PDO extension](https://www.php.net/manual/en/book.pdo.php) for relational database support
- [GraphViz](https://graphviz.org/) for graph visualization
