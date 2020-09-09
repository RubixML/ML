<?php

namespace Rubix\ML\Persisters;

use League;
use League\Flysystem\Adapter\Local;
use League\Flysystem\Adapter\Ftp;
use League\Flysystem\FilesystemInterface;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use RuntimeException;
use Stringable;

/**
 * Flysystem
 *
 * The flysystem persister serializes models to a file at a user-specified path, using a
 * user-provided flysystem instance.
 *
 * Flysystem is a filesystem abstraction providing a unified interface for many different
 * filesystems. (Local, Amazon S3, Azure Blob Storage, Google Cloud Storage, Dropbox etc...)
 *
 * @see https://flysystem.thephpleague.com
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class Flysystem implements Persister, Stringable
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
     * The filesystem implementation providing access to your backend storage.
     *
     * @var \League\Flysystem\FilesystemInterface
     */
    protected $filesystem;

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
     * Local
     *
     * Shortcut to return a Flysystem Persister backed by the Local filesystem
     *
     * @see https://flysystem.thephpleague.com/v1/docs/adapter/local/
     *
     * @param string $path The absolute path to the file on the filesystem.
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @return \Rubix\ML\Persisters\Flysystem
     */
    public static function local(string $path, bool $history = false, ?Serializer $serializer = null) : Flysystem
    {
        $filesystem = new League\Flysystem\Filesystem(new Local(dirname($path)));

        return new Flysystem(basename($path), $filesystem, $history, $serializer);
    }

    /**
     * FTP
     *
     * Shortcut to return a Flysystem Persister backed by an FTP Server
     *
     * @see https://flysystem.thephpleague.com/v1/docs/adapter/ftp/
     *
     * @param string $path
     * @param array<mixed> $config Configuration settings for the FTP adapter.
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @return \Rubix\ML\Persisters\Flysystem
     */
    public static function ftp(string $path, array $config, bool $history = false, ?Serializer $serializer = null) : Flysystem
    {
        $filesystem = new League\Flysystem\Filesystem(new Ftp($config));

        return new Flysystem($path, $filesystem, $history, $serializer);
    }

    /**
     * @param string $path
     * @param FilesystemInterface $filesystem
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     */
    public function __construct(string $path, FilesystemInterface $filesystem, bool $history = false, ?Serializer $serializer = null)
    {
        $this->path = $path;
        $this->filesystem = $filesystem;
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
        if ($this->history and $this->filesystem->has($this->path)) {
            $timestamp = (string) time();

            $filename = $this->path . '-' . $timestamp . '.' . self::HISTORY_EXT;

            $num = 0;
            while ($this->filesystem->has($filename)) {
                $filename = $this->path . '-' . $timestamp . '-' . ++$num . '.' . self::HISTORY_EXT;
            }

            try {
                if (!$this->filesystem->rename($this->path, $filename)) {
                    throw new RuntimeException("Failed to create history file: '{$filename}' {$this}");
                }
            } catch (League\Flysystem\Exception $e) {
                throw new RuntimeException("Failed to create history file: '{$filename}' {$this}");
            }
        }

        $data = $this->serializer->serialize($persistable);
        $success = $this->filesystem->put($this->path, (string) $data);

        if (!$success) {
            throw new RuntimeException("Could not write to filesystem. {$this}");
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
        if (!$this->filesystem->has($this->path)) {
            throw new RuntimeException("File does not exist in filesystem. {$this}");
        }

        $data = new Encoding($this->filesystem->read($this->path) ?: '');

        if ($data->bytes() === 0) {
            throw new RuntimeException("File does not contain any data. {$this}");
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
        $filesystem = 'Flysystem';

        if (method_exists($this->filesystem, 'getAdapter')) {
            $filesystem .= ' <' . Params::toString($this->filesystem->getAdapter()) . '>';
        }

        return "{$filesystem} (path: '{$this->path}', serializer: {$this->serializer})";
    }
}
