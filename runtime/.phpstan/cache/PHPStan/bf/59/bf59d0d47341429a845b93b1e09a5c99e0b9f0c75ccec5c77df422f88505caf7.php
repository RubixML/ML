<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Traits/Multiprocessing.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Traits\Multiprocessing
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-82400f40fa39abc6d0da5c7cbf35d5204da244c51865c49748a793a10da55edc',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Traits\\Multiprocessing',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Traits/Multiprocessing.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Traits',
    'name' => 'Rubix\\ML\\Traits\\Multiprocessing',
    'shortName' => 'Multiprocessing',
    'isInterface' => false,
    'isTrait' => true,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Multiprocessing
 *
 * Multiprocessing is the use of two or more processes that usually execute on
 * multiple cores when training. Estimators that implement the Parallel interface
 * can take advantage of multiple core systems by executing parts or all of the
 * algorithm in parallel.
 *
 * > **Note**: The optimal number of workers will depend on the system
 * specifications of the computer. Fewer workers than CPU cores can result in
 * slower performance but too many workers can cause excess overhead.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 23,
    'endLine' => 41,
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
    ),
    'immediateProperties' => 
    array (
      'backend' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Traits\\Multiprocessing',
        'implementingClassName' => 'Rubix\\ML\\Traits\\Multiprocessing',
        'name' => 'backend',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Backends\\Backend',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The parallel processing backend.
 *
 * @var Backend
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 30,
        'endLine' => 30,
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
      'setBackend' => 
      array (
        'name' => 'setBackend',
        'parameters' => 
        array (
          'backend' => 
          array (
            'name' => 'backend',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Backends\\Backend',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 37,
            'endLine' => 37,
            'startColumn' => 32,
            'endColumn' => 47,
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
 * Set the parallel processing backend.
 *
 * @param Backend $backend
 */',
        'startLine' => 37,
        'endLine' => 40,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Traits\\Multiprocessing',
        'implementingClassName' => 'Rubix\\ML\\Traits\\Multiprocessing',
        'currentClassName' => 'Rubix\\ML\\Traits\\Multiprocessing',
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