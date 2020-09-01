<?php

namespace Rubix\ML\Persisters;

use League\Flysystem\FilesystemInterface;
use Rubix\ML\Encoding;
use Rubix\ML\FilesystemAware;
use Rubix\ML\Other\Traits\FilesystemTrait;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use RuntimeException;
use Stringable;

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
class Filesystem implements FilesystemAware, Persister, Stringable
{
    use FilesystemTrait;

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
     * @param Serializer|null $serializer
     * @param FilesystemInterface|null $filesystem
     */
    public function __construct(string $path, bool $history = false, ?Serializer $serializer = null, ?FilesystemInterface $filesystem = null)
    {
        if ($filesystem) {
            $this->setFilesystem($filesystem);
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
        if ($this->history and $this->filesystem()->has($this->path)) {
            $timestamp = (string) time();

            $filename = $this->path . '-' . $timestamp . '.' . self::HISTORY_EXT;

            $num = 0;

            while ($this->filesystem()->has($filename)) {
                $filename = $this->path . '-' . $timestamp . '-' . ++$num . '.' . self::HISTORY_EXT;
            }

            if (!$this->filesystem()->rename($this->path, $filename)) {
                throw new RuntimeException('Failed to create history file.');
            }
        }

        $this->serializer->serialize($persistable)->write($this->path, $this->filesystem());
    }

    /**
     * Load the last model that was saved.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        if (!$this->filesystem()->has($this->path)) {
            throw new RuntimeException("File {$this->path} does not exist.");
        }

        $data = new Encoding((string) $this->filesystem()->read($this->path));

        if ($data->bytes() === 0) {
            throw new RuntimeException("File {$this->path} does not"
                . ' contain any data.');
        }

        return $this->serializer->unserialize($data);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Filesystem (path: {$this->path},"
            . ' history: ' . Params::toString($this->history) . ','
            . " serializer: {$this->serializer})";
    }
}
