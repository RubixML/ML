<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Unlabeled.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Unlabeled
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-9755202e2c77621e455cc412cfb4bd7372f3fa7ce647ba8106bb2fc56b019608',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Unlabeled.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets',
    'name' => 'Rubix\\ML\\Datasets\\Unlabeled',
    'shortName' => 'Unlabeled',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Unlabeled
 *
 * Unlabeled datasets are used to train unsupervised learners and for feeding unknown
 * samples into an estimator to make predictions during inference. As their name implies,
 * they do not require a corresponding label for each sample.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 31,
    'endLine' => 522,
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
                'startLine' => 39,
                'endLine' => 39,
                'startTokenPos' => 120,
                'startFilePos' => 947,
                'endTokenPos' => 121,
                'endFilePos' => 948,
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
            'startLine' => 39,
            'endLine' => 39,
            'startColumn' => 34,
            'endColumn' => 52,
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
 * Build a new unlabeled dataset with validation.
 *
 * @param array<mixed[]> $samples
 * @return self
 */',
        'startLine' => 39,
        'endLine' => 42,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 50,
                'endLine' => 50,
                'startTokenPos' => 161,
                'startFilePos' => 1202,
                'endTokenPos' => 162,
                'endFilePos' => 1203,
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
            'startLine' => 50,
            'endLine' => 50,
            'startColumn' => 34,
            'endColumn' => 52,
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
 * Build a new unlabeled dataset foregoing validation.
 *
 * @param array<mixed[]> $samples
 * @return self
 */',
        'startLine' => 50,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 61,
            'endLine' => 61,
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
        'startLine' => 61,
        'endLine' => 66,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 75,
            'endLine' => 75,
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
 * @param iterable<Dataset> $datasets
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 75,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 450,
                'startFilePos' => 2938,
                'endTokenPos' => 450,
                'endFilePos' => 2939,
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
            'startLine' => 110,
            'endLine' => 110,
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
        'startLine' => 110,
        'endLine' => 118,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 127,
                'endLine' => 127,
                'startTokenPos' => 520,
                'startFilePos' => 3356,
                'endTokenPos' => 520,
                'endFilePos' => 3357,
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
            'startLine' => 127,
            'endLine' => 127,
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
        'startLine' => 127,
        'endLine' => 135,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 144,
                'endLine' => 144,
                'startTokenPos' => 595,
                'startFilePos' => 3807,
                'endTokenPos' => 595,
                'endFilePos' => 3807,
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
            'startLine' => 144,
            'endLine' => 144,
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
 * Take n samples from this dataset and return them in a new dataset.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 144,
        'endLine' => 152,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 161,
                'endLine' => 161,
                'startTokenPos' => 665,
                'startFilePos' => 4243,
                'endTokenPos' => 665,
                'endFilePos' => 4243,
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
            'startLine' => 161,
            'endLine' => 161,
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
 * Leave n samples on this dataset and return the rest in a new dataset.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 161,
        'endLine' => 169,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 178,
            'endLine' => 178,
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
            'startLine' => 178,
            'endLine' => 178,
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
        'startLine' => 178,
        'endLine' => 181,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 191,
            'endLine' => 191,
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
            'startLine' => 191,
            'endLine' => 191,
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
 * Remove a size n chunk of the dataset starting at offset and return it in
 * a new dataset.
 *
 * @param int $offset
 * @param int $n
 * @return self
 */',
        'startLine' => 191,
        'endLine' => 194,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 203,
            'endLine' => 203,
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
 * Merge another dataset with this dataset.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 203,
        'endLine' => 214,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 223,
            'endLine' => 223,
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
        'startLine' => 223,
        'endLine' => 238,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
        'startLine' => 245,
        'endLine' => 250,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 259,
                'endLine' => 259,
                'startTokenPos' => 1128,
                'startFilePos' => 6951,
                'endTokenPos' => 1128,
                'endFilePos' => 6953,
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
            'startLine' => 259,
            'endLine' => 259,
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
 * Split the dataset into two stratified subsets with a given ratio of samples.
 *
 * @param float $ratio
 * @throws InvalidArgumentException
 * @return array{self,self}
 */',
        'startLine' => 259,
        'endLine' => 272,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
              'code' => '3',
              'attributes' => 
              array (
                'startLine' => 281,
                'endLine' => 281,
                'startTokenPos' => 1266,
                'startFilePos' => 7578,
                'endTokenPos' => 1266,
                'endFilePos' => 7578,
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
            'startLine' => 281,
            'endLine' => 281,
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
        'startLine' => 281,
        'endLine' => 299,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
                'startLine' => 309,
                'endLine' => 309,
                'startTokenPos' => 1404,
                'startFilePos' => 8306,
                'endTokenPos' => 1404,
                'endFilePos' => 8307,
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
            'startLine' => 309,
            'endLine' => 309,
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
 * as many samples as possible.
 *
 * @param positive-int $n
 * @return list<self>
 */',
        'startLine' => 309,
        'endLine' => 312,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 324,
            'endLine' => 324,
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
            'startLine' => 324,
            'endLine' => 324,
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
 * Partition the dataset into left and right subsets using the values of a single feature column for comparison.
 *
 * @internal
 *
 * @param int $column
 * @param string|int|float $value
 * @throws InvalidArgumentException
 * @return array{self,self}
 */',
        'startLine' => 324,
        'endLine' => 347,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 359,
            'endLine' => 359,
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
            'startLine' => 359,
            'endLine' => 359,
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
            'startLine' => 359,
            'endLine' => 359,
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
        'startLine' => 359,
        'endLine' => 375,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 384,
            'endLine' => 384,
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
        'startLine' => 384,
        'endLine' => 407,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 416,
            'endLine' => 416,
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
        'startLine' => 416,
        'endLine' => 432,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 442,
            'endLine' => 442,
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
            'startLine' => 442,
            'endLine' => 442,
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
        'startLine' => 442,
        'endLine' => 494,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
            'startLine' => 504,
            'endLine' => 504,
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
        'startLine' => 503,
        'endLine' => 511,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
        'startLine' => 518,
        'endLine' => 521,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Unlabeled',
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