<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/TokenHashingVectorizer.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\TokenHashingVectorizer
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ca50c360af89e2ebd173ea54ee45dee24468d09b44744049112637e12a8e78f2',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/TokenHashingVectorizer.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
    'shortName' => 'TokenHashingVectorizer',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Token Hashing Vectorizer
 *
 * Token Hashing Vectorizer builds token count vectors on the fly by employing a *hashing
 * trick*. It is a stateless transformer that uses the CRC32 (Cyclic Redundancy Check)
 * hashing algorithm to assign token occurrences to a bucket in a vector of user-defined
 * dimensionality. The advantage of hashing over a fixed vocabulary is that there is no
 * memory footprint however there is a chance that certain tokens will collide with other
 * tokens especially in lower-dimensional vector spaces.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 31,
    'endLine' => 188,
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
      'CRC32' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'CRC32',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'crc32\'',
          'attributes' => 
          array (
            'startLine' => 38,
            'endLine' => 38,
            'startTokenPos' => 91,
            'startFilePos' => 1147,
            'endTokenPos' => 91,
            'endFilePos' => 1153,
          ),
        ),
        'docComment' => '/**
 * The CRC32b callback function.
 *
 * @var callable(string):int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 33,
      ),
      'MURMUR3' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'MURMUR3',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::class, \'murmur3\']',
          'attributes' => 
          array (
            'startLine' => 45,
            'endLine' => 45,
            'startTokenPos' => 104,
            'startFilePos' => 1282,
            'endTokenPos' => 111,
            'endFilePos' => 1305,
          ),
        ),
        'docComment' => '/**
 * The MurmurHash3 callback function.
 *
 * @var callable(string):int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 52,
      ),
      'FNV1' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'FNV1',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::class, \'fnv1\']',
          'attributes' => 
          array (
            'startLine' => 52,
            'endLine' => 52,
            'startTokenPos' => 124,
            'startFilePos' => 1424,
            'endTokenPos' => 131,
            'endFilePos' => 1444,
          ),
        ),
        'docComment' => '/**
 * The FNV1 callback function.
 *
 * @var callable(string):int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
      'MAX_DIMENSIONS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'MAX_DIMENSIONS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2147483647',
          'attributes' => 
          array (
            'startLine' => 59,
            'endLine' => 59,
            'startTokenPos' => 144,
            'startFilePos' => 1575,
            'endTokenPos' => 144,
            'endFilePos' => 1584,
          ),
        ),
        'docComment' => '/**
 * The maximum number of dimensions supported.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 59,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 48,
      ),
    ),
    'immediateProperties' => 
    array (
      'dimensions' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
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
 * The dimensionality of the vector space.
 *
 * @var int<0,max>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 66,
        'endLine' => 66,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'tokenizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'tokenizer',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Tokenizers\\Tokenizer',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The tokenizer used to extract tokens from blobs of text.
 *
 * @var Tokenizer
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 73,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'hashFn' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'name' => 'hashFn',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The hash function that accepts a string token and returns an integer.
 *
 * @var callable(string):int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 80,
        'endLine' => 80,
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
      'murmur3' => 
      array (
        'name' => 'murmur3',
        'parameters' => 
        array (
          'input' => 
          array (
            'name' => 'input',
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
            'startLine' => 88,
            'endLine' => 88,
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
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The 32-bit MurmurHash3 hashing function.
 *
 * @param string $input
 * @return int
 */',
        'startLine' => 88,
        'endLine' => 91,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'aliasName' => NULL,
      ),
      'fnv1' => 
      array (
        'name' => 'fnv1',
        'parameters' => 
        array (
          'input' => 
          array (
            'name' => 'input',
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
            'startLine' => 99,
            'endLine' => 99,
            'startColumn' => 33,
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
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The 32-bit FNV1a hashing function.
 *
 * @param string $input
 * @return int
 */',
        'startLine' => 99,
        'endLine' => 102,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'dimensions' => 
          array (
            'name' => 'dimensions',
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 33,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'tokenizer' => 
          array (
            'name' => 'tokenizer',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 272,
                'startFilePos' => 2733,
                'endTokenPos' => 272,
                'endFilePos' => 2736,
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
                      'name' => 'Rubix\\ML\\Tokenizers\\Tokenizer',
                      'isIdentifier' => false,
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 50,
            'endColumn' => 77,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'hashFn' => 
          array (
            'name' => 'hashFn',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 282,
                'startFilePos' => 2759,
                'endTokenPos' => 282,
                'endFilePos' => 2762,
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
                      'name' => 'callable',
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 80,
            'endColumn' => 103,
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
 * @param int $dimensions
 * @param Tokenizer|null $tokenizer
 * @param callable(string):int|null $hashFn
 * @throws InvalidArgumentException
 */',
        'startLine' => 110,
        'endLine' => 121,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
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
 * @return DataType[]
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
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
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
            'startLine' => 138,
            'endLine' => 138,
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
        'startLine' => 138,
        'endLine' => 141,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'aliasName' => NULL,
      ),
      'vectorize' => 
      array (
        'name' => 'vectorize',
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
            'startLine' => 148,
            'endLine' => 148,
            'startColumn' => 31,
            'endColumn' => 44,
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
 * Vectorize the text features of a sample.
 *
 * @param list<mixed> $sample
 */',
        'startLine' => 148,
        'endLine' => 175,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
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
        'startLine' => 184,
        'endLine' => 187,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TokenHashingVectorizer',
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