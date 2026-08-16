<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Labeled.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Labeled
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-6f3e62e06c1fd3d0a84b4f05740c741c39943c812e849d70cd5e8bce175375e7',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Labeled',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Labeled.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets',
    'name' => 'Rubix\\ML\\Datasets\\Labeled',
    'shortName' => 'Labeled',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Labeled
 *
 * A Labeled dataset is used to train supervised learners and for testing a model by
 * providing the ground-truth. In addition to the standard dataset object methods, a
 * Labeled dataset can perform operations such as stratification and sorting the
 * dataset by label.
 *
 * > **Note:** Labels can be of categorical or continuous data type but NaN values
 * are not allowed.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 44,
    'endLine' => 833,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'labels' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'name' => 'labels',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The observed outcomes for each sample in the dataset.
 *
 * @var list<int|float|string>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
    ),
    'immediateMethods' => 
    array (
      'build' => 
      array (
        'name' => 'build',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 60,
                'endLine' => 60,
                'startTokenPos' => 186,
                'startFilePos' => 1498,
                'endTokenPos' => 187,
                'endFilePos' => 1499,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 60,
            'endLine' => 60,
            'startColumn' => 34,
            'endColumn' => 52,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 60,
                'endLine' => 60,
                'startTokenPos' => 196,
                'startFilePos' => 1518,
                'endTokenPos' => 197,
                'endFilePos' => 1519,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 60,
            'endLine' => 60,
            'startColumn' => 55,
            'endColumn' => 72,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a new labeled dataset with validation.
 *
 * @param array<mixed[]> $samples
 * @param (string|int|float)[] $labels
 * @return self
 */',
        'startLine' => 60,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'quick' => 
      array (
        'name' => 'quick',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 72,
                'endLine' => 72,
                'startTokenPos' => 240,
                'startFilePos' => 1823,
                'endTokenPos' => 241,
                'endFilePos' => 1824,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 72,
            'endLine' => 72,
            'startColumn' => 34,
            'endColumn' => 52,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 72,
                'endLine' => 72,
                'startTokenPos' => 250,
                'startFilePos' => 1843,
                'endTokenPos' => 251,
                'endFilePos' => 1844,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 72,
            'endLine' => 72,
            'startColumn' => 55,
            'endColumn' => 72,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a new labeled dataset foregoing validation.
 *
 * @param array<mixed[]> $samples
 * @param (string|int|float)[] $labels
 * @return self
 */',
        'startLine' => 72,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'fromIterator' => 
      array (
        'name' => 'fromIterator',
        'parameters' => 
        array (
          'iterator' => 
          array (
            'name' => 'iterator',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'iterable',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 83,
            'endLine' => 83,
            'startColumn' => 41,
            'endColumn' => 58,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a dataset with the rows from an iterable data table.
 *
 * @param iterable<mixed[]> $iterator
 * @return self
 */',
        'startLine' => 83,
        'endLine' => 93,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'stack' => 
      array (
        'name' => 'stack',
        'parameters' => 
        array (
          'datasets' => 
          array (
            'name' => 'datasets',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'iterable',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 102,
            'endLine' => 102,
            'startColumn' => 34,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Stack a number of datasets on top of each other to form a single dataset.
 *
 * @param iterable<Labeled> $datasets
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 102,
        'endLine' => 133,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 141,
                'endLine' => 141,
                'startTokenPos' => 605,
                'startFilePos' => 3845,
                'endTokenPos' => 606,
                'endFilePos' => 3846,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 141,
            'endLine' => 141,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 141,
                'endLine' => 141,
                'startTokenPos' => 615,
                'startFilePos' => 3865,
                'endTokenPos' => 616,
                'endFilePos' => 3866,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 141,
            'endLine' => 141,
            'startColumn' => 54,
            'endColumn' => 71,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'verify' => 
          array (
            'name' => 'verify',
            'default' => 
            array (
              'code' => 'true',
              'attributes' => 
              array (
                'startLine' => 141,
                'endLine' => 141,
                'startTokenPos' => 625,
                'startFilePos' => 3884,
                'endTokenPos' => 625,
                'endFilePos' => 3887,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'bool',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 141,
            'endLine' => 141,
            'startColumn' => 74,
            'endColumn' => 92,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param array<mixed[]> $samples
 * @param (string|int|float)[] $labels
 * @param bool $verify
 * @throws InvalidArgumentException
 */',
        'startLine' => 141,
        'endLine' => 176,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'labels' => 
      array (
        'name' => 'labels',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the labels.
 *
 * @return mixed[]
 */',
        'startLine' => 183,
        'endLine' => 186,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'label' => 
      array (
        'name' => 'label',
        'parameters' => 
        array (
          'offset' => 
          array (
            'name' => 'offset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 195,
            'endLine' => 195,
            'startColumn' => 27,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'int',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'float',
                  'isIdentifier' => true,
                ),
              ),
              2 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'string',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return a label at the given row offset.
 *
 * @param int $offset
 * @throws InvalidArgumentException
 * @return int|float|string
 */',
        'startLine' => 195,
        'endLine' => 202,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'labelType' => 
      array (
        'name' => 'labelType',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\DataType',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the integer encoded data type of the label or null if empty.
 *
 * @throws RuntimeException
 * @return DataType
 */',
        'startLine' => 210,
        'endLine' => 217,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'transformLabels' => 
      array (
        'name' => 'transformLabels',
        'parameters' => 
        array (
          'callback' => 
          array (
            'name' => 'callback',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'callable',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 226,
            'endLine' => 226,
            'startColumn' => 37,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Map labels to their new values and return self for method chaining.
 *
 * @param callable $callback
 * @throws RuntimeException
 * @return self
 */',
        'startLine' => 226,
        'endLine' => 240,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'possibleOutcomes' => 
      array (
        'name' => 'possibleOutcomes',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The set of all possible labels.
 *
 * @return mixed[]
 */',
        'startLine' => 247,
        'endLine' => 250,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'head' => 
      array (
        'name' => 'head',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 259,
                'endLine' => 259,
                'startTokenPos' => 1204,
                'startFilePos' => 7187,
                'endTokenPos' => 1204,
                'endFilePos' => 7188,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 259,
            'endLine' => 259,
            'startColumn' => 26,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return a dataset containing only the first n samples.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 259,
        'endLine' => 267,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'tail' => 
      array (
        'name' => 'tail',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 276,
                'endLine' => 276,
                'startTokenPos' => 1274,
                'startFilePos' => 7605,
                'endTokenPos' => 1274,
                'endFilePos' => 7606,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 276,
            'endLine' => 276,
            'startColumn' => 26,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return a dataset containing only the last n samples.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 276,
        'endLine' => 284,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'take' => 
      array (
        'name' => 'take',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => 
            array (
              'code' => '1',
              'attributes' => 
              array (
                'startLine' => 294,
                'endLine' => 294,
                'startTokenPos' => 1349,
                'startFilePos' => 8074,
                'endTokenPos' => 1349,
                'endFilePos' => 8074,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 294,
            'endLine' => 294,
            'startColumn' => 26,
            'endColumn' => 35,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Take n samples and labels from this dataset and return them in a new
 * dataset.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 294,
        'endLine' => 302,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'leave' => 
      array (
        'name' => 'leave',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => 
            array (
              'code' => '1',
              'attributes' => 
              array (
                'startLine' => 312,
                'endLine' => 312,
                'startTokenPos' => 1419,
                'startFilePos' => 8528,
                'endTokenPos' => 1419,
                'endFilePos' => 8528,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 312,
            'endLine' => 312,
            'startColumn' => 27,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Leave n samples and labels on this dataset and return the rest in a new
 * dataset.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 312,
        'endLine' => 320,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'slice' => 
      array (
        'name' => 'slice',
        'parameters' => 
        array (
          'offset' => 
          array (
            'name' => 'offset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 329,
            'endLine' => 329,
            'startColumn' => 27,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'n' => 
          array (
            'name' => 'n',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 329,
            'endLine' => 329,
            'startColumn' => 40,
            'endColumn' => 45,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an n size portion of the dataset in a new dataset.
 *
 * @param int $offset
 * @param int $n
 * @return self
 */',
        'startLine' => 329,
        'endLine' => 335,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'splice' => 
      array (
        'name' => 'splice',
        'parameters' => 
        array (
          'offset' => 
          array (
            'name' => 'offset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 344,
            'endLine' => 344,
            'startColumn' => 28,
            'endColumn' => 38,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'n' => 
          array (
            'name' => 'n',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 344,
            'endLine' => 344,
            'startColumn' => 41,
            'endColumn' => 46,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Remove a size n chunk of the dataset starting at offset and return it in a new dataset.
 *
 * @param int $offset
 * @param int $n
 * @return self
 */',
        'startLine' => 344,
        'endLine' => 350,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'merge' => 
      array (
        'name' => 'merge',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 359,
            'endLine' => 359,
            'startColumn' => 27,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Merge the rows of this dataset with another dataset.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 359,
        'endLine' => 378,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'join' => 
      array (
        'name' => 'join',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 387,
            'endLine' => 387,
            'startColumn' => 26,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Join the columns of this dataset with another dataset.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 387,
        'endLine' => 402,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'randomize' => 
      array (
        'name' => 'randomize',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Randomize the dataset in place and return self for chaining.
 *
 * @return self
 */',
        'startLine' => 409,
        'endLine' => 422,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'stratifyByLabel' => 
      array (
        'name' => 'stratifyByLabel',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Group samples by label and return an array of stratified datasets. i.e.
 * n datasets consisting of samples with the same label where n is equal to
 * the number of unique labels.
 *
 * @return self[]
 */',
        'startLine' => 431,
        'endLine' => 447,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'split' => 
      array (
        'name' => 'split',
        'parameters' => 
        array (
          'ratio' => 
          array (
            'name' => 'ratio',
            'default' => 
            array (
              'code' => '0.5',
              'attributes' => 
              array (
                'startLine' => 456,
                'endLine' => 456,
                'startTokenPos' => 2142,
                'startFilePos' => 12504,
                'endTokenPos' => 2142,
                'endFilePos' => 12506,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 456,
            'endLine' => 456,
            'startColumn' => 27,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Split the dataset into two subsets with a given ratio of samples.
 *
 * @param float $ratio
 * @throws InvalidArgumentException
 * @return array{self,self}
 */',
        'startLine' => 456,
        'endLine' => 476,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'stratifiedSplit' => 
      array (
        'name' => 'stratifiedSplit',
        'parameters' => 
        array (
          'ratio' => 
          array (
            'name' => 'ratio',
            'default' => 
            array (
              'code' => '0.5',
              'attributes' => 
              array (
                'startLine' => 485,
                'endLine' => 485,
                'startTokenPos' => 2309,
                'startFilePos' => 13313,
                'endTokenPos' => 2309,
                'endFilePos' => 13315,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 485,
            'endLine' => 485,
            'startColumn' => 37,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Split the dataset into two stratified subsets with a given ratio of samples.
 *
 * @param float $ratio
 * @throws InvalidArgumentException
 * @return array{self,self}
 */',
        'startLine' => 485,
        'endLine' => 505,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'fold' => 
      array (
        'name' => 'fold',
        'parameters' => 
        array (
          'k' => 
          array (
            'name' => 'k',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 514,
                'endLine' => 514,
                'startTokenPos' => 2473,
                'startFilePos' => 14079,
                'endTokenPos' => 2473,
                'endFilePos' => 14080,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 514,
            'endLine' => 514,
            'startColumn' => 26,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Fold the dataset k - 1 times to form k equal size datasets.
 *
 * @param int $k
 * @throws InvalidArgumentException
 * @return list<self>
 */',
        'startLine' => 514,
        'endLine' => 536,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'stratifiedFold' => 
      array (
        'name' => 'stratifiedFold',
        'parameters' => 
        array (
          'k' => 
          array (
            'name' => 'k',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 545,
                'endLine' => 545,
                'startTokenPos' => 2635,
                'startFilePos' => 14827,
                'endTokenPos' => 2635,
                'endFilePos' => 14828,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 545,
            'endLine' => 545,
            'startColumn' => 36,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Fold the dataset into k equal sized stratified datasets.
 *
 * @param int $k
 * @throws InvalidArgumentException
 * @return list<self>
 */',
        'startLine' => 545,
        'endLine' => 566,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'batch' => 
      array (
        'name' => 'batch',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => 
            array (
              'code' => '50',
              'attributes' => 
              array (
                'startLine' => 576,
                'endLine' => 576,
                'startTokenPos' => 2787,
                'startFilePos' => 15672,
                'endTokenPos' => 2787,
                'endFilePos' => 15673,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 576,
            'endLine' => 576,
            'startColumn' => 27,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a collection of batches of size n from the dataset. If there are
 * not enough samples to fill an entire batch, then the dataset will contain
 * as many samples and labels as possible.
 *
 * @param positive-int $n
 * @return list<self>
 */',
        'startLine' => 576,
        'endLine' => 583,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'splitByFeature' => 
      array (
        'name' => 'splitByFeature',
        'parameters' => 
        array (
          'column' => 
          array (
            'name' => 'column',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 595,
            'endLine' => 595,
            'startColumn' => 36,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'value' => 
          array (
            'name' => 'value',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
              'data' => 
              array (
                'types' => 
                array (
                  0 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'string',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'int',
                      'isIdentifier' => true,
                    ),
                  ),
                  2 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'float',
                      'isIdentifier' => true,
                    ),
                  ),
                ),
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 595,
            'endLine' => 595,
            'startColumn' => 49,
            'endColumn' => 71,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Split the dataset into left and right subsets using the values of a single feature column for comparison.
 *
 * @internal
 *
 * @param int $column
 * @param string|int|float $value
 * @throws InvalidArgumentException
 * @return array{self,self}
 */',
        'startLine' => 595,
        'endLine' => 627,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'spatialSplit' => 
      array (
        'name' => 'spatialSplit',
        'parameters' => 
        array (
          'leftCentroid' => 
          array (
            'name' => 'leftCentroid',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 639,
            'endLine' => 639,
            'startColumn' => 34,
            'endColumn' => 52,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'rightCentroid' => 
          array (
            'name' => 'rightCentroid',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 639,
            'endLine' => 639,
            'startColumn' => 55,
            'endColumn' => 74,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'kernel' => 
          array (
            'name' => 'kernel',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 639,
            'endLine' => 639,
            'startColumn' => 77,
            'endColumn' => 92,
            'parameterIndex' => 2,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Partition the dataset into left and right subsets based on the samples\' distances from two centroids.
 *
 * @internal
 *
 * @param (string|int|float)[] $leftCentroid
 * @param (string|int|float)[] $rightCentroid
 * @param Distance $kernel
 * @return array{self,self}
 */',
        'startLine' => 639,
        'endLine' => 660,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'randomSubset' => 
      array (
        'name' => 'randomSubset',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 669,
            'endLine' => 669,
            'startColumn' => 34,
            'endColumn' => 39,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random subset without replacement.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 669,
        'endLine' => 693,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'randomSubsetWithReplacement' => 
      array (
        'name' => 'randomSubsetWithReplacement',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 702,
            'endLine' => 702,
            'startColumn' => 49,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random subset with replacement.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 702,
        'endLine' => 721,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'randomWeightedSubsetWithReplacement' => 
      array (
        'name' => 'randomWeightedSubsetWithReplacement',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 731,
            'endLine' => 731,
            'startColumn' => 57,
            'endColumn' => 62,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'weights' => 
          array (
            'name' => 'weights',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 731,
            'endLine' => 731,
            'startColumn' => 65,
            'endColumn' => 78,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random weighted subset with replacement.
 *
 * @param int $n
 * @param (int|float)[] $weights
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 731,
        'endLine' => 785,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'describeByLabel' => 
      array (
        'name' => 'describeByLabel',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Report',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Describe the features of the dataset broken down by label.
 *
 * @return Report
 */',
        'startLine' => 792,
        'endLine' => 801,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'offsetGet' => 
      array (
        'name' => 'offsetGet',
        'parameters' => 
        array (
          'offset' => 
          array (
            'name' => 'offset',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 811,
            'endLine' => 811,
            'startColumn' => 31,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'ReturnTypeWillChange',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Return a row from the dataset at the given offset.
 *
 * @param int $offset
 * @throws InvalidArgumentException
 * @return mixed[]
 */',
        'startLine' => 810,
        'endLine' => 818,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
      'getIterator' => 
      array (
        'name' => 'getIterator',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Traversable',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Get an iterator for the samples in the dataset.
 *
 * @return \\Generator<mixed[]>
 */',
        'startLine' => 825,
        'endLine' => 832,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Labeled',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));