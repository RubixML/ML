<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Tokenizers/WordStemmerBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Tokenizers\WordStemmerBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-3f6a661f74f0ce965b2c1d17d47817e9a7e70dd748c27214a5b08a3f7aee23e1',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Tokenizers/WordStemmerBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
    'name' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
    'shortName' => 'WordStemmerBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Tokenizers"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 12,
    'endLine' => 33,
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
      'SAMPLE_TEXT' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'name' => 'SAMPLE_TEXT',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'value' => 
        array (
          'code' => '"Do you see any Teletubbies in here? Do you see a slender plastic tag clipped to my shirt with my name printed on it? Do you see a little Asian child with a blank expression on his face sitting outside on a mechanical helicopter that shakes when you put quarters in it? No? Well, that\'s what you see at a toy store. And you must think you\'re in a toy store, because you\'re here shopping for an infant named Jeb."',
          'attributes' => 
          array (
            'startLine' => 14,
            'endLine' => 14,
            'startTokenPos' => 35,
            'startFilePos' => 248,
            'endTokenPos' => 35,
            'endFilePos' => 659,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 14,
        'endLine' => 14,
        'startColumn' => 5,
        'endColumn' => 454,
      ),
    ),
    'immediateProperties' => 
    array (
      'tokenizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'name' => 'tokenizer',
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
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 16,
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
      'setUp' => 
      array (
        'name' => 'setUp',
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
        'docComment' => NULL,
        'startLine' => 18,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'aliasName' => NULL,
      ),
      'tokenize' => 
      array (
        'name' => 'tokenize',
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
 * @Subject
 * @revs(1000)
 * @Iterations(5)
 * @OutputTimeUnit("milliseconds", precision=3)
 */',
        'startLine' => 29,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\WordStemmerBench',
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