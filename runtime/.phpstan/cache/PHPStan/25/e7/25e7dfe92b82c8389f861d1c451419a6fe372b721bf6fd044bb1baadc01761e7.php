<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Graph/Trees/KDTreeBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Graph\Trees\KDTreeBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-bfda31a22922571fad3a309175759fb4f47424ed0fc59132c2f16a0d4ad24aa8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Graph/Trees/KDTreeBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
    'shortName' => 'KDTreeBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Trees"})
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
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
            'startFilePos' => 284,
            'endTokenPos' => 38,
            'endFilePos' => 288,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'name' => 'dataset',
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
      'tree' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'name' => 'tree',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var KDTree
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 20,
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
        'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'aliasName' => NULL,
      ),
      'grow' => 
      array (
        'name' => 'grow',
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
 * @Iterations(3)
 * @OutputTimeUnit("seconds", precision=3)
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
        'namespace' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Graph\\Trees\\KDTreeBench',
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