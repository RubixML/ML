<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Tokenizers/KSkipNGram.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Tokenizers\KSkipNGram
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-6b9bb88d073c9f891476270151fc02166f2b368bafdb6bf1f1423b3fec21108a',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Tokenizers/KSkipNGram.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Tokenizers',
    'name' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
    'shortName' => 'KSkipNGram',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * K-Skip-N-Gram
 *
 * K-skip-n-grams are a technique similar to n-grams, whereby n-grams are formed but
 * in addition to allowing adjacent sequences of words, the next *k* words will
 * be skipped forming n-grams of the new forward looking sequences. The tokenizer
 * outputs tokens ranging from *min* to *max* number of words per token.
 *
 * References:
 * [1] D. Guthrie et al. A Closer Look at Skip-gram Modelling.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Oksana Yudenko
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 25,
    'endLine' => 146,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Tokenizers\\Tokenizer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'SEPARATOR' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'SEPARATOR',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\' \'',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 48,
            'startFilePos' => 814,
            'endTokenPos' => 48,
            'endFilePos' => 816,
          ),
        ),
        'docComment' => '/**
 * The separator between words in the n-gram.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 36,
      ),
    ),
    'immediateProperties' => 
    array (
      'min' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'min',
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
 * The minimum number of words in a single token.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 39,
        'endLine' => 39,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'max' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'max',
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
 * The maximum number of words in a single token.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 46,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'skip' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'skip',
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
 * The number of words to skip over to form new sequences.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 24,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'wordTokenizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'wordTokenizer',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Tokenizers\\Word',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The word tokenizer.
 *
 * @var Word
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 60,
        'endLine' => 60,
        'startColumn' => 5,
        'endColumn' => 34,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'sentenceTokenizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'name' => 'sentenceTokenizer',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Tokenizers\\Sentence',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The sentence tokenizer.
 *
 * @var Sentence
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 67,
        'endLine' => 67,
        'startColumn' => 5,
        'endColumn' => 42,
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
          'min' => 
          array (
            'name' => 'min',
            'default' => 
            array (
              'code' => '2',
              'attributes' => 
              array (
                'startLine' => 76,
                'endLine' => 76,
                'startTokenPos' => 110,
                'startFilePos' => 1614,
                'endTokenPos' => 110,
                'endFilePos' => 1614,
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
            'startLine' => 76,
            'endLine' => 76,
            'startColumn' => 33,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'max' => 
          array (
            'name' => 'max',
            'default' => 
            array (
              'code' => '2',
              'attributes' => 
              array (
                'startLine' => 76,
                'endLine' => 76,
                'startTokenPos' => 119,
                'startFilePos' => 1628,
                'endTokenPos' => 119,
                'endFilePos' => 1628,
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
            'startLine' => 76,
            'endLine' => 76,
            'startColumn' => 47,
            'endColumn' => 58,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'skip' => 
          array (
            'name' => 'skip',
            'default' => 
            array (
              'code' => '2',
              'attributes' => 
              array (
                'startLine' => 76,
                'endLine' => 76,
                'startTokenPos' => 128,
                'startFilePos' => 1643,
                'endTokenPos' => 128,
                'endFilePos' => 1643,
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
            'startLine' => 76,
            'endLine' => 76,
            'startColumn' => 61,
            'endColumn' => 73,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'wordTokenizer' => 
          array (
            'name' => 'wordTokenizer',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 76,
                'endLine' => 76,
                'startTokenPos' => 138,
                'startFilePos' => 1669,
                'endTokenPos' => 138,
                'endFilePos' => 1672,
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
                      'name' => 'Rubix\\ML\\Tokenizers\\Word',
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
            'startLine' => 76,
            'endLine' => 76,
            'startColumn' => 76,
            'endColumn' => 102,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $min
 * @param int $max
 * @param int $skip
 * @param Word|null $wordTokenizer
 * @throws InvalidArgumentException
 */',
        'startLine' => 76,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'aliasName' => NULL,
      ),
      'tokenize' => 
      array (
        'name' => 'tokenize',
        'parameters' => 
        array (
          'text' => 
          array (
            'name' => 'text',
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
            'startLine' => 104,
            'endLine' => 104,
            'startColumn' => 30,
            'endColumn' => 41,
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
 * Tokenize a blob of text.
 *
 * @param string $text
 * @return list<string>
 */',
        'startLine' => 104,
        'endLine' => 133,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
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
        'startLine' => 142,
        'endLine' => 145,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\KSkipNGram',
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