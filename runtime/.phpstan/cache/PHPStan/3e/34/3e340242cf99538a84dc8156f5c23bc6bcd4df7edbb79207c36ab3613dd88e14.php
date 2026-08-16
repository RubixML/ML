<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Dataset.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Dataset
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-79d721d182591336d4453bb03d4062acd735bb85ed47a0a395e86b470943f5a3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Dataset',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Dataset.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets',
    'name' => 'Rubix\\ML\\Datasets\\Dataset',
    'shortName' => 'Dataset',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 64,
    'docComment' => '/**
 * Dataset
 *
 * In Rubix ML, data are passed in specialized in-memory containers called Dataset
 * objects. Dataset objects are extended table-like data structures with an internal
 * type system and many operations for wrangling. They can hold a heterogeneous mix
 * of categorical and continuous data and they make it easy to transport data in a
 * canonical way.
 *
 * > **Note:** By convention, categorical data are given as string type whereas
 * continuous data are given as either integer or floating point numbers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, array>
 * @implements IteratorAggregate<int, array>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 44,
    'endLine' => 682,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'ArrayAccess',
      1 => 'IteratorAggregate',
      2 => 'Countable',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'samples' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'name' => 'samples',
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
 * The rows of samples and columns of features that make up the
 * data table i.e. the fixed-length feature vectors.
 *
 * @var list<list<mixed>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 29,
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
                'startLine' => 59,
                'endLine' => 59,
                'startTokenPos' => 150,
                'startFilePos' => 1765,
                'endTokenPos' => 151,
                'endFilePos' => 1766,
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
            'startLine' => 59,
            'endLine' => 59,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
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
                'startLine' => 59,
                'endLine' => 59,
                'startTokenPos' => 160,
                'startFilePos' => 1784,
                'endTokenPos' => 160,
                'endFilePos' => 1787,
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
            'startLine' => 59,
            'endLine' => 59,
            'startColumn' => 54,
            'endColumn' => 72,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param mixed[] $samples
 * @param bool $verify
 * @throws InvalidArgumentException
 */',
        'startLine' => 59,
        'endLine' => 93,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 101,
            'endLine' => 101,
            'startColumn' => 50,
            'endColumn' => 67,
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
 * @return static
 */',
        'startLine' => 101,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 76,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 81,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 109,
            'endLine' => 109,
            'startColumn' => 43,
            'endColumn' => 60,
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
 * @return static
 */',
        'startLine' => 109,
        'endLine' => 109,
        'startColumn' => 5,
        'endColumn' => 69,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 81,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'shape' => 
      array (
        'name' => 'shape',
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
 * Return a 2-tuple containing the shape of the sample matrix i.e the number of rows and columns.
 *
 * @return array{int<0,max>,int<0,max>}
 */',
        'startLine' => 116,
        'endLine' => 119,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'size' => 
      array (
        'name' => 'size',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of feature values in the dataset.
 *
 * @return int<0,max>
 */',
        'startLine' => 126,
        'endLine' => 129,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'types' => 
      array (
        'name' => 'types',
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
 * Return the high-level data types of each column in the data table.
 *
 * @return list<DataType>
 */',
        'startLine' => 136,
        'endLine' => 141,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'numSamples' => 
      array (
        'name' => 'numSamples',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of rows in the datasets.
 *
 * @return int<0,max>
 */',
        'startLine' => 148,
        'endLine' => 151,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'sample' => 
      array (
        'name' => 'sample',
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
            'startLine' => 159,
            'endLine' => 159,
            'startColumn' => 28,
            'endColumn' => 38,
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
        ),
        'docComment' => '/**
 * Return the sample at the given row offset.
 *
 * @param int $offset
 * @return list<mixed>
 */',
        'startLine' => 159,
        'endLine' => 166,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'samples' => 
      array (
        'name' => 'samples',
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
 * Return the sample matrix.
 *
 * @return list<list<mixed>>
 */',
        'startLine' => 173,
        'endLine' => 176,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'numFeatures' => 
      array (
        'name' => 'numFeatures',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of feature columns in the dataset.
 *
 * @return int<0,max>
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
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'feature' => 
      array (
        'name' => 'feature',
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
            'startLine' => 194,
            'endLine' => 194,
            'startColumn' => 29,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the feature column at the given offset.
 *
 * @param int $offset
 * @return mixed[]
 */',
        'startLine' => 194,
        'endLine' => 197,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'dropFeature' => 
      array (
        'name' => 'dropFeature',
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
            'startLine' => 205,
            'endLine' => 205,
            'startColumn' => 33,
            'endColumn' => 43,
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
 * Drop a feature column at a given offset from the dataset.
 *
 * @param int $offset
 * @return self
 */',
        'startLine' => 205,
        'endLine' => 212,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'features' => 
      array (
        'name' => 'features',
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
 * Rotate the sample matrix so that the values of each feature become rows.
 *
 * @return mixed[]
 */',
        'startLine' => 219,
        'endLine' => 222,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'featuresByType' => 
      array (
        'name' => 'featuresByType',
        'parameters' => 
        array (
          'type' => 
          array (
            'name' => 'type',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\DataType',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 230,
            'endLine' => 230,
            'startColumn' => 36,
            'endColumn' => 49,
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
        ),
        'docComment' => '/**
 * Return the feature columns that match a given data type.
 *
 * @param DataType $type
 * @return mixed[]
 */',
        'startLine' => 230,
        'endLine' => 241,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'featureType' => 
      array (
        'name' => 'featureType',
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
            'startLine' => 251,
            'endLine' => 251,
            'startColumn' => 33,
            'endColumn' => 43,
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
            'name' => 'Rubix\\ML\\DataType',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Get the data type for a feature column at the given offset.
 *
 * @param int $offset
 * @throws InvalidArgumentException
 * @throws RuntimeException
 * @return DataType
 */',
        'startLine' => 251,
        'endLine' => 265,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'featureTypes' => 
      array (
        'name' => 'featureTypes',
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
 * Return an array of feature column data types autodetected using the first sample in the dataset.
 *
 * @return list<DataType>
 */',
        'startLine' => 272,
        'endLine' => 279,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'uniqueTypes' => 
      array (
        'name' => 'uniqueTypes',
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
 * Return the unique feature types.
 *
 * @return list<DataType>
 */',
        'startLine' => 286,
        'endLine' => 289,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'homogeneous' => 
      array (
        'name' => 'homogeneous',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Do the samples consist of values of a single data type?
 *
 * @return bool
 */',
        'startLine' => 296,
        'endLine' => 299,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'apply' => 
      array (
        'name' => 'apply',
        'parameters' => 
        array (
          'transformer' => 
          array (
            'name' => 'transformer',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Transformers\\Transformer',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 307,
            'endLine' => 307,
            'startColumn' => 27,
            'endColumn' => 50,
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
 * Apply a transformation to the dataset.
 *
 * @param Transformer $transformer
 * @return static
 */',
        'startLine' => 307,
        'endLine' => 318,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'reverseApply' => 
      array (
        'name' => 'reverseApply',
        'parameters' => 
        array (
          'transformer' => 
          array (
            'name' => 'transformer',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Transformers\\Reversible',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 327,
            'endLine' => 327,
            'startColumn' => 34,
            'endColumn' => 56,
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
 * Reverse a transformation that was applied to the dataset.
 *
 * @param Reversible $transformer
 * @throws RuntimeException
 * @return static
 */',
        'startLine' => 327,
        'endLine' => 338,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'filter' => 
      array (
        'name' => 'filter',
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
            'startLine' => 346,
            'endLine' => 346,
            'startColumn' => 28,
            'endColumn' => 45,
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
 * Filter the records of the dataset using a callback function to determine if a row should be included in the return dataset.
 *
 * @param callable $callback
 * @return static
 */',
        'startLine' => 346,
        'endLine' => 349,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'describe' => 
      array (
        'name' => 'describe',
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
 * Return an array of statistics such as the central tendency, dispersion
 * and shape of each continuous feature column and the joint probabilities
 * of every categorical feature column.
 *
 * @throws RuntimeException
 * @return Report
 */',
        'startLine' => 359,
        'endLine' => 426,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'sort' => 
      array (
        'name' => 'sort',
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
            'startLine' => 435,
            'endLine' => 435,
            'startColumn' => 26,
            'endColumn' => 43,
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
 * Sort the records in the dataset using a callback for comparisons between samples. The callback function
 * accepts two records to be compared and should return `true` if the records should be swapped.
 *
 * @param callable $callback
 * @return static
 */',
        'startLine' => 435,
        'endLine' => 462,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'deduplicate' => 
      array (
        'name' => 'deduplicate',
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
 * Remove duplicate rows from the dataset.
 *
 * @return self
 */',
        'startLine' => 469,
        'endLine' => 472,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'exportTo' => 
      array (
        'name' => 'exportTo',
        'parameters' => 
        array (
          'extractor' => 
          array (
            'name' => 'extractor',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Extractors\\Exporter',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 480,
            'endLine' => 480,
            'startColumn' => 30,
            'endColumn' => 48,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'overwrite' => 
          array (
            'name' => 'overwrite',
            'default' => 
            array (
              'code' => 'false',
              'attributes' => 
              array (
                'startLine' => 480,
                'endLine' => 480,
                'startTokenPos' => 2068,
                'startFilePos' => 13204,
                'endTokenPos' => 2068,
                'endFilePos' => 13208,
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
            'startLine' => 480,
            'endLine' => 480,
            'startColumn' => 51,
            'endColumn' => 73,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Write the dataset to the location and format given by a writable extractor.
 *
 * @param Exporter $extractor
 * @param bool $overwrite
 */',
        'startLine' => 480,
        'endLine' => 483,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'empty' => 
      array (
        'name' => 'empty',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Is the dataset empty?
 *
 * @return bool
 */',
        'startLine' => 490,
        'endLine' => 493,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 501,
                'endLine' => 501,
                'startTokenPos' => 2134,
                'startFilePos' => 13606,
                'endTokenPos' => 2134,
                'endFilePos' => 13607,
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
            'startLine' => 501,
            'endLine' => 501,
            'startColumn' => 35,
            'endColumn' => 45,
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
 * @return static
 */',
        'startLine' => 501,
        'endLine' => 501,
        'startColumn' => 5,
        'endColumn' => 54,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 509,
                'endLine' => 509,
                'startTokenPos' => 2158,
                'startFilePos' => 13788,
                'endTokenPos' => 2158,
                'endFilePos' => 13789,
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
            'startLine' => 509,
            'endLine' => 509,
            'startColumn' => 35,
            'endColumn' => 45,
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
 * @return static
 */',
        'startLine' => 509,
        'endLine' => 509,
        'startColumn' => 5,
        'endColumn' => 54,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 517,
                'endLine' => 517,
                'startTokenPos' => 2182,
                'startFilePos' => 13983,
                'endTokenPos' => 2182,
                'endFilePos' => 13983,
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
            'startLine' => 517,
            'endLine' => 517,
            'startColumn' => 35,
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
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Take n samples from the dataset and return them in a new dataset.
 *
 * @param int $n
 * @return static
 */',
        'startLine' => 517,
        'endLine' => 517,
        'startColumn' => 5,
        'endColumn' => 53,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 525,
                'endLine' => 525,
                'startTokenPos' => 2206,
                'startFilePos' => 14181,
                'endTokenPos' => 2206,
                'endFilePos' => 14181,
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
            'startLine' => 525,
            'endLine' => 525,
            'startColumn' => 36,
            'endColumn' => 45,
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
 * Leave n samples on the dataset and return the rest in a new dataset.
 *
 * @param int $n
 * @return static
 */',
        'startLine' => 525,
        'endLine' => 525,
        'startColumn' => 5,
        'endColumn' => 54,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 534,
            'endLine' => 534,
            'startColumn' => 36,
            'endColumn' => 46,
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
            'startLine' => 534,
            'endLine' => 534,
            'startColumn' => 49,
            'endColumn' => 54,
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
 * @return static
 */',
        'startLine' => 534,
        'endLine' => 534,
        'startColumn' => 5,
        'endColumn' => 63,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 544,
            'endLine' => 544,
            'startColumn' => 37,
            'endColumn' => 47,
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
            'startLine' => 544,
            'endLine' => 544,
            'startColumn' => 50,
            'endColumn' => 55,
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
 * @return static
 */',
        'startLine' => 544,
        'endLine' => 544,
        'startColumn' => 5,
        'endColumn' => 64,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 552,
            'endLine' => 552,
            'startColumn' => 36,
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
 * Merge another dataset with this dataset.
 *
 * @param Dataset $dataset
 * @return static
 */',
        'startLine' => 552,
        'endLine' => 552,
        'startColumn' => 5,
        'endColumn' => 60,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 560,
            'endLine' => 560,
            'startColumn' => 35,
            'endColumn' => 50,
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
 * @return static
 */',
        'startLine' => 560,
        'endLine' => 560,
        'startColumn' => 5,
        'endColumn' => 59,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 568,
                'endLine' => 568,
                'startTokenPos' => 2320,
                'startFilePos' => 15265,
                'endTokenPos' => 2320,
                'endFilePos' => 15267,
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
            'startLine' => 568,
            'endLine' => 568,
            'startColumn' => 36,
            'endColumn' => 53,
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
 * @return array{self,self}
 */',
        'startLine' => 568,
        'endLine' => 568,
        'startColumn' => 5,
        'endColumn' => 63,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 576,
                'endLine' => 576,
                'startTokenPos' => 2344,
                'startFilePos' => 15460,
                'endTokenPos' => 2344,
                'endFilePos' => 15461,
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
            'startColumn' => 35,
            'endColumn' => 45,
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
 * @return list<self>
 */',
        'startLine' => 576,
        'endLine' => 576,
        'startColumn' => 5,
        'endColumn' => 55,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
                'startLine' => 586,
                'endLine' => 586,
                'startTokenPos' => 2368,
                'startFilePos' => 15786,
                'endTokenPos' => 2368,
                'endFilePos' => 15787,
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
            'startLine' => 586,
            'endLine' => 586,
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
 * Generate a collection of batches of size n from the dataset. If there are
 * not enough samples to fill an entire batch, then the dataset will contain
 * as many samples as possible.
 *
 * @param int $n
 * @return list<self>
 */',
        'startLine' => 586,
        'endLine' => 586,
        'startColumn' => 5,
        'endColumn' => 56,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'splitByFeature' => 
      array (
        'name' => 'splitByFeature',
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
            'startLine' => 597,
            'endLine' => 597,
            'startColumn' => 45,
            'endColumn' => 55,
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
            'startLine' => 597,
            'endLine' => 597,
            'startColumn' => 58,
            'endColumn' => 80,
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
 * @param int $offset
 * @param string|int|float $value
 * @return array{self,self}
 */',
        'startLine' => 597,
        'endLine' => 597,
        'startColumn' => 5,
        'endColumn' => 90,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 609,
            'endLine' => 609,
            'startColumn' => 43,
            'endColumn' => 61,
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
            'startLine' => 609,
            'endLine' => 609,
            'startColumn' => 64,
            'endColumn' => 83,
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
            'startLine' => 609,
            'endLine' => 609,
            'startColumn' => 86,
            'endColumn' => 101,
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
        'startLine' => 609,
        'endLine' => 609,
        'startColumn' => 5,
        'endColumn' => 103,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
 * Randomize the dataset.
 *
 * @return static
 */',
        'startLine' => 616,
        'endLine' => 616,
        'startColumn' => 5,
        'endColumn' => 48,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 624,
            'endLine' => 624,
            'startColumn' => 43,
            'endColumn' => 48,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random subset without replacement.
 *
 * @param int $n
 * @return self
 */',
        'startLine' => 624,
        'endLine' => 624,
        'startColumn' => 5,
        'endColumn' => 50,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 632,
            'endLine' => 632,
            'startColumn' => 58,
            'endColumn' => 63,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random subset of n samples with replacement.
 *
 * @param int $n
 * @return self
 */',
        'startLine' => 632,
        'endLine' => 632,
        'startColumn' => 5,
        'endColumn' => 65,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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
            'startLine' => 641,
            'endLine' => 641,
            'startColumn' => 66,
            'endColumn' => 71,
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
            'startLine' => 641,
            'endLine' => 641,
            'startColumn' => 74,
            'endColumn' => 87,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate a random weighted subset with replacement.
 *
 * @param int $n
 * @param (int|float)[] $weights
 * @return self
 */',
        'startLine' => 641,
        'endLine' => 641,
        'startColumn' => 5,
        'endColumn' => 89,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 65,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'count' => 
      array (
        'name' => 'count',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of rows in the dataset.
 *
 * @return int
 */',
        'startLine' => 648,
        'endLine' => 651,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'offsetSet' => 
      array (
        'name' => 'offsetSet',
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
            'startLine' => 658,
            'endLine' => 658,
            'startColumn' => 31,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'values' => 
          array (
            'name' => 'values',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 658,
            'endLine' => 658,
            'startColumn' => 40,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $offset
 * @param mixed[] $values
 * @throws RuntimeException
 */',
        'startLine' => 658,
        'endLine' => 661,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'offsetExists' => 
      array (
        'name' => 'offsetExists',
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
            'startLine' => 669,
            'endLine' => 669,
            'startColumn' => 34,
            'endColumn' => 40,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Does a given row exist in the dataset.
 *
 * @param int $offset
 * @return bool
 */',
        'startLine' => 669,
        'endLine' => 672,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'aliasName' => NULL,
      ),
      'offsetUnset' => 
      array (
        'name' => 'offsetUnset',
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
            'startLine' => 678,
            'endLine' => 678,
            'startColumn' => 33,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $offset
 * @throws RuntimeException
 */',
        'startLine' => 678,
        'endLine' => 681,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Dataset',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Dataset',
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