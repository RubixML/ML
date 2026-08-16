<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/SafeEuclidean.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Kernels\Distance\SafeEuclidean
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-851af6f5d6ee7ef5a9cefef75996deb372c9900a2e436b46faa801f04a45c958',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/SafeEuclidean.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Kernels\\Distance',
    'name' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
    'shortName' => 'SafeEuclidean',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Safe Euclidean
 *
 * An Euclidean distance metric suitable for samples that may contain NaN
 * (not a number) values i.e. missing data. The Safe Euclidean metric approximates
 * the Euclidean distance function by dropping NaN values and scaling the distance
 * according to the proportion of non-NaNs (in either a or b or both) to compensate.
 *
 * References:
 * [1] J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 26,
    'endLine' => 94,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Kernels\\Distance\\Distance',
      1 => 'Rubix\\ML\\Kernels\\Distance\\BoxPrunable',
      2 => 'Rubix\\ML\\Kernels\\Distance\\NaNSafe',
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
 * Return the data types that this kernel is compatible with.
 *
 * @internal
 *
 * @return list<DataType>
 */',
        'startLine' => 35,
        'endLine' => 40,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'aliasName' => NULL,
      ),
      'compute' => 
      array (
        'name' => 'compute',
        'parameters' => 
        array (
          'a' => 
          array (
            'name' => 'a',
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 29,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'b' => 
          array (
            'name' => 'b',
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 39,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the distance between two vectors.
 *
 * @internal
 *
 * @param list<int|float> $a
 * @param list<int|float> $b
 * @return float
 */',
        'startLine' => 51,
        'endLine' => 81,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the string representation of the object.
 *
 * @internal
 *
 * @return string
 */',
        'startLine' => 90,
        'endLine' => 93,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\SafeEuclidean',
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