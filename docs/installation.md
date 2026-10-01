# Installation

Rubix ML installs into your project using [Composer](https://getcomposer.org/) and has optional extensions that can be installed via [PIE](https://github.com/php/pie).

## Requirements

- [PHP](https://php.net/manual/en/install.php) 8.3 or above.

### Recommended

- [Tensor Ext 4.0+](https://github.com/RubixML/Tensor) for fast Matrix/Vector computing.
- [Swoole extension](https://openswoole.com/) for fast multiprocessing support.

### Optional

- [GD extension](https://php.net/manual/en/book.image.php) for image support.
- [Mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for fast multibyte string manipulation.
- [SVM extension](https://php.net/manual/en/book.svm.php) for Support Vector Machine engine (libsvm).
- [PDO extension](https://www.php.net/manual/en/book.pdo.php) for relational database support.
- [GraphViz](https://graphviz.org/) for graph visualization.

## Example

Install Rubix ML into your project using [Composer](https://getcomposer.org/):

```sh
composer require rubix/ml
```

Install the recommended extensions using [PIE](https://github.com/php/pie):

```sh
pie install rubix/tensor_ext swoole/swoole
```
