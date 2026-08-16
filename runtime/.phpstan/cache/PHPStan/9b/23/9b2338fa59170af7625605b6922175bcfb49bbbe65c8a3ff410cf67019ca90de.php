<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/Hamming.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Kernels\Distance\Hamming
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-f421bd2141e7cadd2e243d5d8c9063ed8525e4854cf96399e7ede77d39dc353e',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/Hamming.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Kernels\\Distance',
    'name' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
    'shortName' => 'Hamming',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Hamming
 *
 * A categorical distance function that measures distance as the number of
 * substitutions necessary to convert one sample to the other.
 *
 * References:
 * [1] R. W. Hamming. (1950). Error detecting and error correcting codes.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 20,
    'endLine' => 69,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Kernels\\Distance\\Distance',
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
        'startLine' => 29,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
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
            'startLine' => 45,
            'endLine' => 45,
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
            'startLine' => 45,
            'endLine' => 45,
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
 * @param list<string> $a
 * @param list<string> $b
 * @return float
 */',
        'startLine' => 45,
        'endLine' => 56,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
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
        'startLine' => 65,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Hamming',
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