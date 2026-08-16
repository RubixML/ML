<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Tokenizers/NGramBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Tokenizers\NGramBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-b4ae7431cef7ff7384cb550a5ad4205fc8e51c1aa8d0390ad79d821d3cac0750',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Tokenizers/NGramBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
    'name' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
    'shortName' => 'NGramBench',
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
    'startLine' => 11,
    'endLine' => 32,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
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
            'startLine' => 13,
            'endLine' => 13,
            'startTokenPos' => 30,
            'startFilePos' => 206,
            'endTokenPos' => 30,
            'endFilePos' => 617,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 13,
        'endLine' => 13,
        'startColumn' => 5,
        'endColumn' => 454,
      ),
    ),
    'immediateProperties' => 
    array (
      'tokenizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'name' => 'tokenizer',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Tokenizers\\NGram',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 31,
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
        'startLine' => 17,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
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
        'startLine' => 28,
        'endLine' => 31,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Tokenizers\\NGramBench',
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