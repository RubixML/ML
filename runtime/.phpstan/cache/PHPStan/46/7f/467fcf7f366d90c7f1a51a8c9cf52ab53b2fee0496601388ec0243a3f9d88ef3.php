<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/L2NormalizerBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Transformers\L2NormalizerBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-f7d478c37a6bfe4084052229e336ba4bf2f29cab200ec455e8e913dc36b383c5',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/L2NormalizerBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
    'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
    'shortName' => 'L2NormalizerBench',
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
    'endLine' => 49,
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
      'DATASET_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'name' => 'DATASET_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 15,
            'endLine' => 15,
            'startTokenPos' => 38,
            'startFilePos' => 305,
            'endTokenPos' => 38,
            'endFilePos' => 309,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
    ),
    'immediateProperties' => 
    array (
      'dataset' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'name' => 'dataset',
        'modifiers' => 1,
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
        'endColumn' => 20,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'transformer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'name' => 'transformer',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var L2Normalizer
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 27,
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
        'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
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
 * @Iterations(5)
 * @OutputTimeUnit("milliseconds", precision=3)
 */',
        'startLine' => 45,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\L2NormalizerBench',
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