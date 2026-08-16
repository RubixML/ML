<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/ImageRotator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\ImageRotator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-80374afbb576a505ae06915149636d7f27b009497f95a1b05990869acc2a59c5',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/ImageRotator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\ImageRotator',
    'shortName' => 'ImageRotator',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Randomized Image Rotator
 *
 * Randomly rotates the image between 0 and a given number of max degrees.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 24,
    'endLine' => 168,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Transformers\\Transformer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'FILL_COLOR' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'name' => 'FILL_COLOR',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 31,
            'endLine' => 31,
            'startTokenPos' => 65,
            'startFilePos' => 712,
            'endTokenPos' => 65,
            'endFilePos' => 712,
          ),
        ),
        'docComment' => '/**
 * The color of the area of the image filled in after rotation and cropping.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 31,
        'endLine' => 31,
        'startColumn' => 5,
        'endColumn' => 35,
      ),
    ),
    'immediateProperties' => 
    array (
      'offset' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'name' => 'offset',
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
 * The number of degrees to rotate the image.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'jitter' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'name' => 'jitter',
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
 * The amount of random jitter to add to the rotation.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 28,
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
          'offset' => 
          array (
            'name' => 'offset',
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
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 33,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'jitter' => 
          array (
            'name' => 'jitter',
            'default' => 
            array (
              'code' => '0.0',
              'attributes' => 
              array (
                'startLine' => 52,
                'endLine' => 52,
                'startTokenPos' => 105,
                'startFilePos' => 1142,
                'endTokenPos' => 105,
                'endFilePos' => 1144,
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
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 48,
            'endColumn' => 66,
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
 * @param float $offset
 * @param float $jitter
 * @throws InvalidArgumentException
 */',
        'startLine' => 52,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
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
 * Return the data types that this transformer is compatible with.
 *
 * @internal
 *
 * @return list<DataType>
 */',
        'startLine' => 77,
        'endLine' => 80,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'aliasName' => NULL,
      ),
      'transform' => 
      array (
        'name' => 'transform',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 87,
            'endLine' => 87,
            'startColumn' => 31,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Transform the dataset in place.
 *
 * @param array<mixed[]> $samples
 */',
        'startLine' => 87,
        'endLine' => 90,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'aliasName' => NULL,
      ),
      'rotateAndCrop' => 
      array (
        'name' => 'rotateAndCrop',
        'parameters' => 
        array (
          'sample' => 
          array (
            'name' => 'sample',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 99,
            'endLine' => 99,
            'startColumn' => 38,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Randomly rotates and crops the images in a sample to their original size.
 *
 * @internal
 *
 * @param list<mixed> $sample
 */',
        'startLine' => 99,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'aliasName' => NULL,
      ),
      'rotationAngle' => 
      array (
        'name' => 'rotationAngle',
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
 * Return an angle with a given offset with random jitter in degrees between 0 and 360.
 *
 * @return float
 */',
        'startLine' => 134,
        'endLine' => 155,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
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
        'startLine' => 164,
        'endLine' => 167,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
        'currentClassName' => 'Rubix\\ML\\Transformers\\ImageRotator',
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