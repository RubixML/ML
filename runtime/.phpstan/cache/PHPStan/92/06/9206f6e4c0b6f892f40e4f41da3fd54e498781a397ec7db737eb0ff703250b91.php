<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../andrewdalpino/okbloomer/src/BloomFilter.php-PHPStan\BetterReflection\Reflection\ReflectionClass-OkBloomer\BloomFilter
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-b5027c9938c7aa35befb49a682b45efe371265e412d688594af96cf2a3414c70-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'OkBloomer\\BloomFilter',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../andrewdalpino/okbloomer/src/BloomFilter.php',
      ),
    ),
    'namespace' => 'OkBloomer',
    'name' => 'OkBloomer\\BloomFilter',
    'shortName' => 'BloomFilter',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Bloom Filter
 *
 * A probabilistic data structure that estimates the prior occurrence of a given item with a maximum false positive rate.
 *
 * References:
 * [1] P. S. Almeida et al. (2007). Scalable Bloom Filters.
 *
 * @category    Data Structures
 * @package     Scienide/OkBloomer
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 26,
    'endLine' => 363,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'MAX_32_BIT_INTEGER' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'MAX_32_BIT_INTEGER',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2147483647',
          'attributes' => 
          array (
            'startLine' => 33,
            'endLine' => 33,
            'startTokenPos' => 72,
            'startFilePos' => 659,
            'endTokenPos' => 72,
            'endFilePos' => 668,
          ),
        ),
        'docComment' => '/**
 * The maximum 32 bit integer.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 33,
        'endLine' => 33,
        'startColumn' => 5,
        'endColumn' => 52,
      ),
    ),
    'immediateProperties' => 
    array (
      'maxFalsePositiveRate' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'maxFalsePositiveRate',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The false positive rate to remain below.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 40,
        'endLine' => 40,
        'startColumn' => 5,
        'endColumn' => 36,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'numHashes' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'numHashes',
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
 * The number of hash functions used, i.e. the number of slices per layer.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 47,
        'endLine' => 47,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'layerSize' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'layerSize',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The size of each layer of the filter in bits.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 54,
        'endLine' => 54,
        'startColumn' => 5,
        'endColumn' => 25,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'sliceSize' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'sliceSize',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The size of each slice of each layer in bits.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 61,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 25,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'layers' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'layers',
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
 * The layers of the filter.
 *
 * @var list<\\OkBloomer\\BooleanArray>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 68,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'm' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'm',
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
 * The size of the filter in bits.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 75,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 21,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'n' => 
      array (
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'name' => 'n',
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
        'default' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 82,
            'endLine' => 82,
            'startTokenPos' => 133,
            'startFilePos' => 1519,
            'endTokenPos' => 133,
            'endFilePos' => 1519,
          ),
        ),
        'docComment' => '/**
 * The number of items in the filter.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 82,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 25,
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
          'maxFalsePositiveRate' => 
          array (
            'name' => 'maxFalsePositiveRate',
            'default' => 
            array (
              'code' => '0.01',
              'attributes' => 
              array (
                'startLine' => 91,
                'endLine' => 91,
                'startTokenPos' => 151,
                'startFilePos' => 1777,
                'endTokenPos' => 151,
                'endFilePos' => 1780,
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
            'startLine' => 91,
            'endLine' => 91,
            'startColumn' => 9,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'numHashes' => 
          array (
            'name' => 'numHashes',
            'default' => 
            array (
              'code' => '4',
              'attributes' => 
              array (
                'startLine' => 92,
                'endLine' => 92,
                'startTokenPos' => 161,
                'startFilePos' => 1809,
                'endTokenPos' => 161,
                'endFilePos' => 1809,
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
                      'name' => 'int',
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
            'startLine' => 92,
            'endLine' => 92,
            'startColumn' => 9,
            'endColumn' => 27,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'layerSize' => 
          array (
            'name' => 'layerSize',
            'default' => 
            array (
              'code' => '32000000',
              'attributes' => 
              array (
                'startLine' => 93,
                'endLine' => 93,
                'startTokenPos' => 170,
                'startFilePos' => 1837,
                'endTokenPos' => 170,
                'endFilePos' => 1844,
              ),
            ),
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
            'startLine' => 93,
            'endLine' => 93,
            'startColumn' => 9,
            'endColumn' => 33,
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
 * @param float $maxFalsePositiveRate
 * @param int|null $numHashes
 * @param int $layerSize
 * @throws \\OkBloomer\\Exceptions\\InvalidArgumentException
 */',
        'startLine' => 90,
        'endLine' => 128,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'maxFalsePositiveRate' => 
      array (
        'name' => 'maxFalsePositiveRate',
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
 * Return the maximum false positive rate of the filter.
 *
 * @return float
 */',
        'startLine' => 135,
        'endLine' => 138,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'numHashes' => 
      array (
        'name' => 'numHashes',
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
 * Return the number of hash functions used in the filter.
 *
 * @return int
 */',
        'startLine' => 145,
        'endLine' => 148,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'layerSize' => 
      array (
        'name' => 'layerSize',
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
 * Return the size of each layer of the filter.
 *
 * @return int
 */',
        'startLine' => 155,
        'endLine' => 158,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'sliceSize' => 
      array (
        'name' => 'sliceSize',
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
 * Return the size of a slice of a layer in bits.
 *
 * @return int
 */',
        'startLine' => 165,
        'endLine' => 168,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'numLayers' => 
      array (
        'name' => 'numLayers',
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
 * Return the number of layers in the filter.
 *
 * @return int
 */',
        'startLine' => 175,
        'endLine' => 178,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'size' => 
      array (
        'name' => 'size',
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
 * Return the size of the Bloom filter in bits.
 *
 * @return int
 */',
        'startLine' => 185,
        'endLine' => 188,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'n' => 
      array (
        'name' => 'n',
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
 * Return the number of bits that are set in the filter.
 *
 * @return int
 */',
        'startLine' => 195,
        'endLine' => 198,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'utilization' => 
      array (
        'name' => 'utilization',
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
 * Return the proportion of bits that are set.
 *
 * @return float
 */',
        'startLine' => 205,
        'endLine' => 208,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'capacity' => 
      array (
        'name' => 'capacity',
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
 * Return the proportion of bits that are not set.
 *
 * @return float
 */',
        'startLine' => 215,
        'endLine' => 218,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'falsePositiveRate' => 
      array (
        'name' => 'falsePositiveRate',
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
 * Return the probability of a recording a false positive.
 *
 * @return float
 */',
        'startLine' => 225,
        'endLine' => 228,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'insert' => 
      array (
        'name' => 'insert',
        'parameters' => 
        array (
          'token' => 
          array (
            'name' => 'token',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 235,
            'endLine' => 235,
            'startColumn' => 28,
            'endColumn' => 40,
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
 * Insert an element into the filter.
 *
 * @param string $token
 */',
        'startLine' => 235,
        'endLine' => 259,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'existsOrInsert' => 
      array (
        'name' => 'existsOrInsert',
        'parameters' => 
        array (
          'token' => 
          array (
            'name' => 'token',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 267,
            'endLine' => 267,
            'startColumn' => 36,
            'endColumn' => 48,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Does a token exist in the filter? If so, return true or insert and return false.
 *
 * @param string $token
 * @return bool
 */',
        'startLine' => 267,
        'endLine' => 307,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'exists' => 
      array (
        'name' => 'exists',
        'parameters' => 
        array (
          'token' => 
          array (
            'name' => 'token',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 315,
            'endLine' => 315,
            'startColumn' => 28,
            'endColumn' => 40,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Does a token exist in the filter?
 *
 * @param string $token
 * @return bool
 */',
        'startLine' => 315,
        'endLine' => 330,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'addLayer' => 
      array (
        'name' => 'addLayer',
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
 * Add a layer to the filter.
 */',
        'startLine' => 335,
        'endLine' => 340,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
        'aliasName' => NULL,
      ),
      'hashes' => 
      array (
        'name' => 'hashes',
        'parameters' => 
        array (
          'token' => 
          array (
            'name' => 'token',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 348,
            'endLine' => 348,
            'startColumn' => 31,
            'endColumn' => 43,
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
 * Return an array of hashes from a given token.
 *
 * @param string $token
 * @return list<int>
 */',
        'startLine' => 348,
        'endLine' => 362,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'OkBloomer',
        'declaringClassName' => 'OkBloomer\\BloomFilter',
        'implementingClassName' => 'OkBloomer\\BloomFilter',
        'currentClassName' => 'OkBloomer\\BloomFilter',
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