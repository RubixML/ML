<?php

namespace Rubix\Engine\Persisters;

use InvalidArgumentException;
use RuntimeException;
use Redis;

class PHPRedis implements Persister
{
    const DEFAULT_OPTIONS = [
        'password' => null,
        'timeout' => 1,
        'delay' => 100,
    ];

    /**
     * The key that identifies the model in the database.
     *
     * @var string
     */
    protected $key;

    /**
     * The Redis connection.
     *
     * @var \Redis
     */
    protected $connection;

    /**
     * @param  string  $key
     * @param  string  $host
     * @param  int  $port
     * @param  int  $database
     * @param  array  $options
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function __construct(string $key, string $host, int $port = 6379, int $database = 0, array $options = [])
    {
        if (!extension_loaded('redis')) {
            throw new RuntimeException('The Redis PHP extension is not loaded.');
        }

        $options = array_replace(self::DEFAULT_OPTIONS, $options);

        $this->connection = new Redis();

        $this->connection->connect($host, $port, $options['timeout'], $options['delay']);
        $this->connection->select($database);

        if (isset($options['password'])) {
            $this->connection->auth($options['password']);
        }

        $this->key = $key;
    }

    /**
     * @param  \Rubix\Engine\Persisters\Persistable  $persistable
     * @return bool
     */
    public function save(Persistable $persistable) : bool
    {
        if ((strlen($persistable) * 1e-6) > 512) {
            throw new RuntimeException('Redis cannot handle models larger than 512 megabytes.');
        }

        return $this->connection->set($this->key, serialize($persistable));
    }

    /**
     * @throws \RuntimeException
     * @return \Rubix\Engine\Persistable|null
     */
    public function load() : Persistable
    {
        $persistable = $this->connection->get($this->key);

        if ($persistable === false) {
            throw new RuntimeException('Could not load object from the database.');
        }

        $persistable = unserialize($persistable);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Object could not be reconstituted.');
        }

        return $persistable;
    }
}
