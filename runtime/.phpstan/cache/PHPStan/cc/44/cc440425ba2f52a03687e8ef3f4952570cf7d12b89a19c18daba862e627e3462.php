<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/TSNEBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Embedders\TSNEBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-f5223c5804398da385fea4da6fc72750f44b75fdba9da860d26c407d248a29ee',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/TSNEBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Embedders',
    'name' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
    'shortName' => 'TSNEBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 13,
    'endLine' => 50,
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
      'TESTING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'name' => 'TESTING_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1000',
          'attributes' => 
          array (
            'startLine' => 15,
            'endLine' => 15,
            'startTokenPos' => 38,
            'startFilePos' => 286,
            'endTokenPos' => 38,
            'endFilePos' => 289,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 40,
      ),
    ),
    'immediateProperties' => 
    array (
      'testing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'name' => 'testing',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var \\Rubix\\ML\\Datasets\\Labeled
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'embedder' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'name' => 'embedder',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var TSNE
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 24,
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
        'startLine' => 27,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Embedders',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'aliasName' => NULL,
      ),
      'apply' => 
      array (
        'name' => 'apply',
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
 * @Skip
 * @Iterations(5)
 * @OutputTimeUnit("seconds", precision=3)
 */',
        'startLine' => 46,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Embedders',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Embedders\\TSNEBench',
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