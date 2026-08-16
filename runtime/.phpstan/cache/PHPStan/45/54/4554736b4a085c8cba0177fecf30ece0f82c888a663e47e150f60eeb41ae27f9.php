<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Hyperplane.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Generators\Hyperplane
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-b456a330a4a065758b3fa03ea419f4b84e95c85f01d883edbcad68ee633ea7f0',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Hyperplane.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets\\Generators',
    'name' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
    'shortName' => 'Hyperplane',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Hyperplane
 *
 * Generates a labeled dataset whose samples form a hyperplane in n-dimensional vector
 * space and whose labels are continuous values drawn from a uniform random distribution
 * between -1 and 1. When the number of coefficients is either 1, 2 or 3, the samples
 * form points, lines, and planes respectively. Due to its linearity, Hyperplane is
 * especially useful for testing linear regression models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 23,
    'endLine' => 110,
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
      'coefficients' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'name' => 'coefficients',
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
 * The n coefficients of the hyperplane where n is the dimensionality.
 *
 * @var Vector
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 30,
        'endLine' => 30,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'intercept' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'name' => 'intercept',
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
 * The y intercept term.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'noise' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
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
 * The factor of gaussian noise to add to the data points.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
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
          'coefficients' => 
          array (
            'name' => 'coefficients',
            'default' => 
            array (
              'code' => '[1, -1]',
              'attributes' => 
              array (
                'startLine' => 53,
                'endLine' => 53,
                'startTokenPos' => 81,
                'startFilePos' => 1339,
                'endTokenPos' => 87,
                'endFilePos' => 1345,
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
            'startLine' => 53,
            'endLine' => 53,
            'startColumn' => 9,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'intercept' => 
          array (
            'name' => 'intercept',
            'default' => 
            array (
              'code' => '0.0',
              'attributes' => 
              array (
                'startLine' => 54,
                'endLine' => 54,
                'startTokenPos' => 96,
                'startFilePos' => 1375,
                'endTokenPos' => 96,
                'endFilePos' => 1377,
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
            'startLine' => 54,
            'endLine' => 54,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 1,
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
                'startLine' => 55,
                'endLine' => 55,
                'startTokenPos' => 105,
                'startFilePos' => 1403,
                'endTokenPos' => 105,
                'endFilePos' => 1405,
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
            'startLine' => 55,
            'endLine' => 55,
            'startColumn' => 9,
            'endColumn' => 26,
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
 * @param (int|float)[] $coefficients
 * @param float $intercept
 * @param float $noise
 * @throws InvalidArgumentException
 */',
        'startLine' => 52,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
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
        'startLine' => 79,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
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
            'startLine' => 90,
            'endLine' => 90,
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
        'startLine' => 90,
        'endLine' => 109,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Hyperplane',
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