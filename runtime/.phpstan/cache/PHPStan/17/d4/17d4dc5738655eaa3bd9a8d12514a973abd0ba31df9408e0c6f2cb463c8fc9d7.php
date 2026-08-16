<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Estimator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Estimator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-84a6766603ca3505fff87223d1c772b06288be945d25d7f70e5a5d884fac50da',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Estimator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Estimator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML',
    'name' => 'Rubix\\ML\\Estimator',
    'shortName' => 'Estimator',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Estimator
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 15,
    'endLine' => 51,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Stringable',
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
      'type' => 
      array (
        'name' => 'type',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\EstimatorType',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the estimator type.
 *
 * @internal
 *
 * @return EstimatorType
 */',
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 43,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\Estimator',
        'implementingClassName' => 'Rubix\\ML\\Estimator',
        'currentClassName' => 'Rubix\\ML\\Estimator',
        'aliasName' => NULL,
      ),
      'compatibility' => 
      array (
        'name' => 'compatibility',
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
 * Return the data types that the estimator is compatible with.
 *
 * @internal
 *
 * @return list<DataType>
 */',
        'startLine' => 33,
        'endLine' => 33,
        'startColumn' => 5,
        'endColumn' => 44,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\Estimator',
        'implementingClassName' => 'Rubix\\ML\\Estimator',
        'currentClassName' => 'Rubix\\ML\\Estimator',
        'aliasName' => NULL,
      ),
      'params' => 
      array (
        'name' => 'params',
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
 * Return the settings of the hyper-parameters in an associative array.
 *
 * @internal
 *
 * @return mixed[]
 */',
        'startLine' => 42,
        'endLine' => 42,
        'startColumn' => 5,
        'endColumn' => 37,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\Estimator',
        'implementingClassName' => 'Rubix\\ML\\Estimator',
        'currentClassName' => 'Rubix\\ML\\Estimator',
        'aliasName' => NULL,
      ),
      'predict' => 
      array (
        'name' => 'predict',
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
            'startLine' => 50,
            'endLine' => 50,
            'startColumn' => 29,
            'endColumn' => 44,
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
 * Make predictions from a dataset.
 *
 * @param Dataset $dataset
 * @return list<string|int|float>
 */',
        'startLine' => 50,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 54,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\Estimator',
        'implementingClassName' => 'Rubix\\ML\\Estimator',
        'currentClassName' => 'Rubix\\ML\\Estimator',
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