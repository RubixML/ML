<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/Canberra.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Kernels\Distance\Canberra
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1b9a343eb89ef8738ddf00fe5b61fc2b6b66c269e27ecc0871413d4103d4b1eb',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Kernels/Distance/Canberra.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Kernels\\Distance',
    'name' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
    'shortName' => 'Canberra',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Canberra
 *
 * A weighted version of the Manhattan distance, Canberra examines the sum of
 * a series of fractional differences between two samples. Canberra can be
 * very sensitive when both coordinates are near zero.
 *
 * References:
 * [1] G. N. Lance et al. (1967). Mixed-data classificatory programs I.
 * Agglomerative Systems.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 24,
    'endLine' => 74,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Kernels\\Distance\\Distance',
      1 => 'Rubix\\ML\\Kernels\\Distance\\BoxPrunable',
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
        'startLine' => 33,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
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
            'startLine' => 49,
            'endLine' => 49,
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
            'startLine' => 49,
            'endLine' => 49,
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
        'startLine' => 49,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
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
        'startLine' => 70,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Kernels\\Distance',
        'declaringClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'implementingClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
        'currentClassName' => 'Rubix\\ML\\Kernels\\Distance\\Canberra',
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