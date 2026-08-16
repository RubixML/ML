<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Nodes/VantagePoint.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Nodes\VantagePoint
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-8372ee2831dc1057876c4d0498a72324e11aabdd967c30783d0d6f7eeef45436',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Nodes/VantagePoint.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Nodes',
    'name' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
    'shortName' => 'VantagePoint',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Vantage Point
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 21,
    'endLine' => 160,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Graph\\Nodes\\Hypersphere',
      1 => 'Rubix\\ML\\Graph\\Nodes\\HasBinaryChildren',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'center' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'name' => 'center',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The center or multivariate mean of the centroid.
 *
 * @var list<string|int|float>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 30,
        'endLine' => 30,
        'startColumn' => 5,
        'endColumn' => 22,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'radius' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'name' => 'radius',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The radius of the centroid.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 22,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'subsets' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'name' => 'subsets',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The left and right splits of the training data.
 *
 * @var array{Labeled,Labeled}|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 23,
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
      'split' => 
      array (
        'name' => 'split',
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
                'name' => 'Rubix\\ML\\Datasets\\Labeled',
                'isIdentifier' => false,
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
            'startColumn' => 34,
            'endColumn' => 49,
            'parameterIndex' => 0,
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
            'startLine' => 53,
            'endLine' => 53,
            'startColumn' => 52,
            'endColumn' => 67,
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
 * Factory method to build a hypersphere by splitting the dataset into left and right clusters.
 *
 * @param Labeled $dataset
 * @param Distance $kernel
 * @return self
 */',
        'startLine' => 53,
        'endLine' => 94,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'center' => 
          array (
            'name' => 'center',
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
            'startLine' => 101,
            'endLine' => 101,
            'startColumn' => 33,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'radius' => 
          array (
            'name' => 'radius',
            'default' => NULL,
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
            'startLine' => 101,
            'endLine' => 101,
            'startColumn' => 48,
            'endColumn' => 60,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'subsets' => 
          array (
            'name' => 'subsets',
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
            'startLine' => 101,
            'endLine' => 101,
            'startColumn' => 63,
            'endColumn' => 76,
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
 * @param list<string|int|float> $center
 * @param float $radius
 * @param array{Labeled,Labeled} $subsets
 */',
        'startLine' => 101,
        'endLine' => 106,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      'center' => 
      array (
        'name' => 'center',
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
 * Return the center vector.
 *
 * @return list<string|int|float>
 */',
        'startLine' => 113,
        'endLine' => 116,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      'radius' => 
      array (
        'name' => 'radius',
        'parameters' => 
        array (
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
 * Return the radius of the centroid.
 *
 * @return float
 */',
        'startLine' => 123,
        'endLine' => 126,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      'subsets' => 
      array (
        'name' => 'subsets',
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
 * Return the left and right subsets of the training data.
 *
 * @throws RuntimeException
 * @return array{Labeled,Labeled}
 */',
        'startLine' => 134,
        'endLine' => 141,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      'isPoint' => 
      array (
        'name' => 'isPoint',
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
 * Does the hypersphere reduce to a single point?
 *
 * @return bool
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
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'aliasName' => NULL,
      ),
      'cleanup' => 
      array (
        'name' => 'cleanup',
        'parameters' => 
        array (
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
 * Remove the left and right splits of the training data.
 */',
        'startLine' => 156,
        'endLine' => 159,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\VantagePoint',
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