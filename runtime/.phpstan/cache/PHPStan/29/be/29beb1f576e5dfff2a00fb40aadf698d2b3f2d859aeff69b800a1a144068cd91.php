<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/HalfMoon.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Generators\HalfMoon
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-483a7da9348db57ab7691fd87e76a0f34e81065a564e9f0138a86c58e456f156',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/HalfMoon.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets\\Generators',
    'name' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
    'shortName' => 'HalfMoon',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Half Moon
 *
 * Generates a dataset consisting of 2-d samples that form the shape of a half moon when
 * plotted on a scatter plot chart.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 22,
    'endLine' => 129,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Datasets\\Generators\\Generator',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'center' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'name' => 'center',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Tensor\\Vector',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The center vector of the circle.
 *
 * @var Vector
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 29,
        'endLine' => 29,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'scale' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'name' => 'scale',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The scaling factor of the half moon.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 36,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'rotation' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'name' => 'rotation',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The rotation on the half moon in degrees.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 43,
        'endLine' => 43,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'noise' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'name' => 'noise',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The factor of Gaussian noise to add to the data points.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 50,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 27,
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
          'x' => 
          array (
            'name' => 'x',
            'default' => 
            array (
              'code' => '0.0',
              'attributes' => 
              array (
                'startLine' => 61,
                'endLine' => 61,
                'startTokenPos' => 97,
                'startFilePos' => 1207,
                'endTokenPos' => 97,
                'endFilePos' => 1209,
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
            'startLine' => 61,
            'endLine' => 61,
            'startColumn' => 9,
            'endColumn' => 22,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'y' => 
          array (
            'name' => 'y',
            'default' => 
            array (
              'code' => '0.0',
              'attributes' => 
              array (
                'startLine' => 62,
                'endLine' => 62,
                'startTokenPos' => 106,
                'startFilePos' => 1231,
                'endTokenPos' => 106,
                'endFilePos' => 1233,
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
            'startLine' => 62,
            'endLine' => 62,
            'startColumn' => 9,
            'endColumn' => 22,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'scale' => 
          array (
            'name' => 'scale',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 63,
                'endLine' => 63,
                'startTokenPos' => 115,
                'startFilePos' => 1259,
                'endTokenPos' => 115,
                'endFilePos' => 1261,
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
            'startLine' => 63,
            'endLine' => 63,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'rotation' => 
          array (
            'name' => 'rotation',
            'default' => 
            array (
              'code' => '90.0',
              'attributes' => 
              array (
                'startLine' => 64,
                'endLine' => 64,
                'startTokenPos' => 124,
                'startFilePos' => 1290,
                'endTokenPos' => 124,
                'endFilePos' => 1293,
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
            'startLine' => 64,
            'endLine' => 64,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'noise' => 
          array (
            'name' => 'noise',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 65,
                'endLine' => 65,
                'startTokenPos' => 133,
                'startFilePos' => 1319,
                'endTokenPos' => 133,
                'endFilePos' => 1321,
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
            'startLine' => 65,
            'endLine' => 65,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param float $x
 * @param float $y
 * @param float $scale
 * @param float $rotation
 * @param float $noise
 * @throws InvalidArgumentException
 */',
        'startLine' => 60,
        'endLine' => 86,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'aliasName' => NULL,
      ),
      'dimensions' => 
      array (
        'name' => 'dimensions',
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
 * Return the dimensionality of the data this generates.
 *
 * @internal
 *
 * @return int<0,max>
 */',
        'startLine' => 95,
        'endLine' => 98,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'aliasName' => NULL,
      ),
      'generate' => 
      array (
        'name' => 'generate',
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
            'startLine' => 106,
            'endLine' => 106,
            'startColumn' => 30,
            'endColumn' => 35,
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
            'name' => 'Rubix\\ML\\Datasets\\Labeled',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate n data points.
 *
 * @param int<0,max> $n
 * @return Labeled
 */',
        'startLine' => 106,
        'endLine' => 128,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\HalfMoon',
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