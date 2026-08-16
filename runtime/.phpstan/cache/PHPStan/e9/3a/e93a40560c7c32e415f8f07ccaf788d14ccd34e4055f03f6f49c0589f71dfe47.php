<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/MinMaxNormalizerBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Transformers\MinMaxNormalizerBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1dfab3840e2ff7876469303a25d2f44b3310da589ffc9dce137209ad13dd9c3d',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Transformers/MinMaxNormalizerBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Transformers',
    'name' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
    'shortName' => 'MinMaxNormalizerBench',
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
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
            'startFilePos' => 313,
            'endTokenPos' => 38,
            'endFilePos' => 317,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'name' => 'transformer',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var MinMaxNormalizer
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Transformers\\MinMaxNormalizerBench',
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