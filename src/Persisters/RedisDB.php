<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Persisters\Serializers\Serializer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Redis;

/**
 * Redis DB
 *
 * Redis is a high performance in-memory key value store that can be used to persist models over a network.
 *
 * > **Note**: Requires the PHP Redis extension and a properly configured Redis server.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RedisDB implements Persister
{
    /**
     * The key of the model in storage.
     *
     * @var string
     */
    protected $key;

    /**
     * The connector to the Redis database.
     *
     * @var \Redis
     */
    protected $db;

    /**
     * The serializer used to convert to and from serial format.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $serializer;

    /**
     * @param string $key
     * @param string $host
     * @param int $port
     * @param int $database
     * @param string|null $password
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @param float $timeout
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function __construct(
        string $key,
        string $host = '127.0.0.1',
        int $port = 6379,
        int $database = 0,
        ?string $password = null,
        ?Serializer $serializer = null,
        float $timeout = 2.5
    ) {
        ExtensionIsLoaded::with('redis')->check();

        if (empty($key)) {
            throw new InvalidArgumentException('Key cannot be empty.');
        }

        if ($timeout <= 0.0) {
            throw new InvalidArgumentException('Timeout must be greater than'
                . " 0, $timeout given.");
        }

        $db = new Redis();

        if (!$db->connect($host, $port, $timeout)) {
            throw new RuntimeException('Could not connect to Redis server'
                . " at host $host on port $port.");
        }

        if (isset($password)) {
            if (!$db->auth($password)) {
                throw new RuntimeException('Password is invalid.');
            }
        }

        if (!$db->select($database)) {
            throw new RuntimeException("Failed to select database $database.");
        }

        $this->key = $key;
        $this->db = $db;
        $this->serializer = $serializer ?? new Native();
    }

    /**
     * Save the persistable object.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function save(Persistable $persistable) : void
    {
        $data = $this->serializer->serialize($persistable);

        $success = $this->db->set($this->key, $data);

        if (!$success) {
            throw new RuntimeException('Failed to save '
                . ' persistable to the database.');
        }
    }

    /**
     * Load the last saved persistable instance.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        $data = new Encoding($this->db->get($this->key) ?: '');

        return $this->serializer->unserialize($data);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Redis DB (key: {$this->key}, serializer: {$this->serializer})";
    }
}
