<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Helpers/CPU.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Helpers\CPU
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-2f0a7dbce770b7c871315951c578350147905fff7b6fb742cbea0860d50290b8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Helpers\\CPU',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Helpers/CPU.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Helpers',
    'name' => 'Rubix\\ML\\Helpers\\CPU',
    'shortName' => 'CPU',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * CPU
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 18,
    'endLine' => 87,
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
      'WIN_CORES' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'name' => 'WIN_CORES',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'wmic cpu get NumberOfCores\'',
          'attributes' => 
          array (
            'startLine' => 25,
            'endLine' => 25,
            'startTokenPos' => 37,
            'startFilePos' => 391,
            'endTokenPos' => 37,
            'endFilePos' => 418,
          ),
        ),
        'docComment' => '/**
 * The command to return the number of processor cores on Windows OS.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 61,
      ),
      'CPU_INFO' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'name' => 'CPU_INFO',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/proc/cpuinfo\'',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 50,
            'startFilePos' => 572,
            'endTokenPos' => 50,
            'endFilePos' => 586,
          ),
        ),
        'docComment' => '/**
 * The command to return the number of processor cores on Linux.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 47,
      ),
      'CORE_REGEX' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'name' => 'CORE_REGEX',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/^processor/m\'',
          'attributes' => 
          array (
            'startLine' => 39,
            'endLine' => 39,
            'startTokenPos' => 63,
            'startFilePos' => 735,
            'endTokenPos' => 63,
            'endFilePos' => 749,
          ),
        ),
        'docComment' => '/**
 * The regular expression used to extract the core count.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 39,
        'endLine' => 39,
        'startColumn' => 5,
        'endColumn' => 49,
      ),
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'cores' => 
      array (
        'name' => 'cores',
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
 * Return the number of cpu cores or 0 if unable to detect.
 *
 * @throws RuntimeException
 * @return int
 */',
        'startLine' => 47,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Helpers',
        'declaringClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'currentClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'aliasName' => NULL,
      ),
      'epsilon' => 
      array (
        'name' => 'epsilon',
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
 * Return the estimated machine epsilon.
 *
 * @return float
 */',
        'startLine' => 75,
        'endLine' => 86,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Helpers',
        'declaringClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\CPU',
        'currentClassName' => 'Rubix\\ML\\Helpers\\CPU',
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