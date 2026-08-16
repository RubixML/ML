<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Blob.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Datasets\Generators\Blob
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ed70136c415a8371b3847b9cd9836ba18a989fb41202c2b193fde9ecdb7061a6',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Datasets/Generators/Blob.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Datasets\\Generators',
    'name' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
    'shortName' => 'Blob',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Blob
 *
 * A normally distributed n-dimensional blob of samples centered at a given
 * mean vector. The standard deviation can be set for the whole blob or for each
 * feature column independently. When a global standard deviation is used, the
 * resulting blob will be isotropic and will converge asymptotically to a sphere.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 28,
    'endLine' => 148,
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
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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
 * The center vector of the blob.
 *
 * @var Vector
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 35,
        'endLine' => 35,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'stdDev' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'name' => 'stdDev',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The standard deviation of the blob.
 *
 * @var Vector|int|float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 42,
        'endLine' => 42,
        'startColumn' => 5,
        'endColumn' => 22,
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
      'simulate' => 
      array (
        'name' => 'simulate',
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 37,
            'endColumn' => 52,
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
 * Fit a Blob generator to the samples in a dataset.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 51,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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
            'default' => 
            array (
              'code' => '[0, 0]',
              'attributes' => 
              array (
                'startLine' => 77,
                'endLine' => 77,
                'startTokenPos' => 249,
                'startFilePos' => 1936,
                'endTokenPos' => 254,
                'endFilePos' => 1941,
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
            'startLine' => 77,
            'endLine' => 77,
            'startColumn' => 33,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'stdDev' => 
          array (
            'name' => 'stdDev',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 77,
                'endLine' => 77,
                'startTokenPos' => 261,
                'startFilePos' => 1954,
                'endTokenPos' => 261,
                'endFilePos' => 1956,
              ),
            ),
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 77,
            'endLine' => 77,
            'startColumn' => 57,
            'endColumn' => 69,
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
 * @param (int|float)[] $center
 * @param int|float|(int|float)[] $stdDev
 * @throws InvalidArgumentException
 */',
        'startLine' => 77,
        'endLine' => 107,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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
 * Return the center coordinates of the Blob.
 *
 * @return list<int|float>
 */',
        'startLine' => 114,
        'endLine' => 117,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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
        'startLine' => 126,
        'endLine' => 129,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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
            'startLine' => 137,
            'endLine' => 137,
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
            'name' => 'Rubix\\ML\\Datasets\\Unlabeled',
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
 * @return Unlabeled
 */',
        'startLine' => 137,
        'endLine' => 147,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Datasets\\Generators',
        'declaringClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'implementingClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
        'currentClassName' => 'Rubix\\ML\\Datasets\\Generators\\Blob',
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