<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Filesystem
 *
 * Filesystems are local or remote storage drives that are organized by files
 * and folders. The filesystem persister serializes models to a file at a
 * user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Filesystem implements Persister
{
    /**
     * The extension to give files created as part of a persistable's save history.
     *
     * @var string
     */
    public const HISTORY_EXT = 'old';

    /**
     * The path to the model file on the filesystem.
     *
     * @var string
     */
    protected $path;

    /**
     * Should we keep a history of past saves?
     *
     * @var bool
     */
    protected $history;

    /**
     * The serializer used to convert to and from serial format.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $serializer;

    /**
     * @param string $path
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @throws \InvalidArgumentException
     */
    public function __construct(string $path, bool $history = false, ?Serializer $serializer = null)
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or'
                . ' is not writable, check path and permissions.');
        }

        $this->path = $path;
        $this->history = $history;
        $this->serializer = $serializer ?? new Native();
    }

    /**
     * Save the persistable model.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \RuntimeException
     */
    public function save(Persistable $persistable) : void
    {
        if ($this->history and is_file($this->path)) {
            $timestamp = (string) time();
            
            $filename = $this->path . '-' . $timestamp . '.' . self::HISTORY_EXT;

            $num = 0;

            while (file_exists($filename)) {
                $filename = $this->path . '-' . $timestamp . '-' . ++$num . '.' . self::HISTORY_EXT;
            }

            if (!rename($this->path, $filename)) {
                throw new RuntimeException('Failed to create history file.');
            }
        }

        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("File $this->path is"
                . ' not writable.');
        }

        $data = $this->serializer->serialize($persistable);

        if (!file_put_contents($this->path, $data, LOCK_EX)) {
            throw new RuntimeException('Failed to write data to'
                . ' the filesystem');
        }
    }

    /**
     * Load the last model that was saved.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        if (!is_readable($this->path)) {
            throw new RuntimeException("File $this->path is not readable.");
        }
            
        $data = file_get_contents($this->path) ?: '';

        if (empty($data)) {
            throw new RuntimeException("File $this->path does not"
                . ' contain any data.');
        }

        $persistable = $this->serializer->unserialize($data);

        return $persistable;
    }
}
