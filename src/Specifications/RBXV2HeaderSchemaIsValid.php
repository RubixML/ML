<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\RuntimeException;

use function is_string;
use function is_array;
use function is_int;

/**
 * @internal
 */
class RBXV2HeaderSchemaIsValid extends Specification
{
    /**
     * The message header.
     *
     * @var array<string, mixed>
     */
    protected array $header;

    /**
     * Build a specification object with the given arguments.
     *
     * @param array<string, mixed> $header
     * @return self
     */
    public static function with(array $header) : self
    {
        return new self($header);
    }

    /**
     * @param array<string, mixed> $header
     * @throws RuntimeException
     */
    public function __construct(array $header)
    {
        $this->header = $header;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws RuntimeException
     */
    public function check() : void
    {
        if (!isset($this->header['class']['name'])) {
            throw new RuntimeException('Header is missing class name.');
        }

        if (!is_string($this->header['class']['name'])) {
            throw new RuntimeException('Class name must be a string.');
        }

        if (!isset($this->header['class']['allowed'])) {
            throw new RuntimeException('Header is missing allowed classes.');
        }

        if (!is_array($this->header['class']['allowed'])) {
            throw new RuntimeException('Allowed classes must be an array.');
        }

        if (!isset($this->header['class']['revision'])) {
            throw new RuntimeException('Header is missing class revision.');
        }

        if (!is_string($this->header['class']['revision'])) {
            throw new RuntimeException('Class revision must be a string.');
        }

        if (!isset($this->header['data']['checksum']['type'])) {
            throw new RuntimeException('Header is missing checksum type.');
        }

        if (!is_string($this->header['data']['checksum']['type'])) {
            throw new RuntimeException('Checksum type must be a string.');
        }

        if (!isset($this->header['data']['checksum']['hash'])) {
            throw new RuntimeException('Header is missing checksum hash.');
        }

        if (!is_string($this->header['data']['checksum']['hash'])) {
            throw new RuntimeException('Checksum hash must be a string.');
        }

        if (!isset($this->header['data']['length'])) {
            throw new RuntimeException('Header is missing data length.');
        }

        if (!is_int($this->header['data']['length'])) {
            throw new RuntimeException('Data length must be an integer.');
        }
    }
}
