<?php

namespace Rubix\ML\Other\Helpers;

use League\Flysystem\Adapter\Local;
use League\Flysystem\AdapterInterface;
use League\Flysystem\Filesystem;
use League\Flysystem\FilesystemInterface;
use League\Flysystem\Memory\MemoryAdapter;
use League\Flysystem\Adapter\Ftp;

/**
 * Storage
 *
 * A helper class providing functions relating to storage/persistence.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Storage
{
    /**
     * Returns a Filesystem instance backed by the provided filesystem.
     *
     *
     * @param AdapterInterface $adapter
     * @param array<mixed> $config
     *
     * @return FilesystemInterface
     */
    public static function filesystem(AdapterInterface $adapter, array $config = []) : FilesystemInterface
    {
        return new Filesystem($adapter, $config);
    }

    /**
     * Returns a Filesystem instance backed by the local filesystem.
     *
     * @param string $root
     * @param array<mixed> $config
     *
     * @return FilesystemInterface
     */
    public static function local(string $root = '/', array $config = []) : FilesystemInterface
    {
        return self::filesystem(new Local($root), $config);
    }

    /**
     * Returns a Filesystem instance backed by a remote FTP server.
     *
     * @param array<mixed> $ftpConfig
     * @param array<mixed> $config
     *
     * @return FilesystemInterface
     */
    public static function ftp(array $ftpConfig, array $config = []) : FilesystemInterface
    {
        return self::filesystem(new Ftp($ftpConfig), $config);
    }

    /**
     * Returns a Filesystem instance backed by an in-memory mock filesystem.
     * Suitable for use in tests or when long-term persistence isn't required.
     *
     * @param array<mixed> $config
     * @return FilesystemInterface
     */
    public static function memory(array $config = []) : FilesystemInterface
    {
        return self::filesystem(new MemoryAdapter(), $config);
    }
}
