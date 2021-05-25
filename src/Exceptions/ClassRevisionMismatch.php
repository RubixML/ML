<?php

namespace Rubix\ML\Exceptions;

use function version_compare;

use const Rubix\ML\VERSION;

class ClassRevisionMismatch extends RuntimeException
{
    /**
     * The version number of the library that the incompatible object was created with.
     *
     * @var string
     */
    protected string $createdWithVersion;

    /**
     * @param string $createdWithVersion
     */
    public function __construct(string $createdWithVersion)
    {
        $direction = version_compare($createdWithVersion, VERSION) >= 0 ? 'up' : 'down';

        parent::__construct('Object incompatible with class revision,'
            . " {$direction}grade to version $createdWithVersion.");

        $this->createdWithVersion = $createdWithVersion;
    }

    /**
     * Return the version number of the library that the incompatible object was created with.
     *
     * @return string
     */
    public function createdWithVersion() : string
    {
        return $this->createdWithVersion;
    }
}
