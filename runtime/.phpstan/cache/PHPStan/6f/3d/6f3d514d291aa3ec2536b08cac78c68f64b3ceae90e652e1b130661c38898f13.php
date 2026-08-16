<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Agglomerate.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Generators\Agglomerate
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1b9b98450be8a1ca74bb8e3566e4fd9e5f9f1783cb99b72cfe4dfab54a05d335',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Agglomerate.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets\\Generators',
    'name' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
    'shortName' => 'Agglomerate',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Agglomerate
 *
 * An Agglomerate is a collection of other generators each assigned with a
 * user-definable label. Agglomerates are useful for classification,
 * clustering, and anomaly detection problems where the label is a discrete
 * value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 22,
    'endLine' => 159,
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
      'generators' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'name' => 'generators',
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
 * An array of generators.
 *
 * @var Generator[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 29,
        'endLine' => 29,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'weights' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'name' => 'weights',
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
 * The normalized weights of each generator i.e. the probability that a
 * sample from a particular generator shows up in the dataset.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'dimensions' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'name' => 'dimensions',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The dimensionality of the agglomerate.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 30,
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
          'generators' => 
          array (
            'name' => 'generators',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 51,
                'endLine' => 51,
                'startTokenPos' => 77,
                'startFilePos' => 1174,
                'endTokenPos' => 78,
                'endFilePos' => 1175,
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 33,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'weights' => 
          array (
            'name' => 'weights',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 51,
                'endLine' => 51,
                'startTokenPos' => 88,
                'startFilePos' => 1196,
                'endTokenPos' => 88,
                'endFilePos' => 1199,
              ),
            ),
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
                      'name' => 'array',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'null',
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 57,
            'endColumn' => 78,
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
 * @param Generator[] $generators
 * @param (int|float)[]|null $weights
 * @throws InvalidArgumentException
 */',
        'startLine' => 51,
        'endLine' => 109,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'aliasName' => NULL,
      ),
      'weights' => 
      array (
        'name' => 'weights',
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
 * Return the normalized weights of each generator in the agglomerate.
 *
 * @return (int|float)[]
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
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
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
 * @return int
 */',
        'startLine' => 128,
        'endLine' => 131,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
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
            'startLine' => 139,
            'endLine' => 139,
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
 * @param int $n
 * @return Labeled
 */',
        'startLine' => 139,
        'endLine' => 158,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Agglomerate',
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