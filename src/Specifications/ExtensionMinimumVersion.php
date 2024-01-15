<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\RuntimeException;

use function phpversion;
use function version_compare;

/**
 * @internal
 */
class ExtensionMinimumVersion extends Specification
{
    /**
     * The name of the extension under consideration.
     *
     * @var string
     */
    protected string $name;

    /**
     * The minimum version of the extension.
     *
     * @var string
     */
    protected string $minVersion;

    /**
     * Build a specification object with the given arguments.
     *
     * @param string $name
     * @param string $minVersion
     * @return self
     */
    public static function with(string $name, string $minVersion) : self
    {
        return new self($name, $minVersion);
    }

    /**
     * @param string $name
     * @param string $minVersion
     */
    public function __construct(string $name, string $minVersion)
    {
        $this->name = $name;
        $this->minVersion = $minVersion;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws RuntimeException
     */
    public function check() : void
    {
        $version = phpversion($this->name);

        if (!$version) {
            throw new RuntimeException("Version number for {$this->name} not available.");
        }

        if (version_compare($version, $this->minVersion, '<')) {
            throw new RuntimeException("The {$this->name} extension version must be"
                . " greater than {$this->minVersion}, $version given.");
        }
    }
}
