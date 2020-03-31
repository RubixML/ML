<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use InvalidArgumentException;
use RuntimeException;
use Redis;

/**
 * Redis DB
 *
 * Redis is a high performance in-memory key value store that can be used to
 * persist models over a network.
 *
 * > **Note**: Requires the PHP Redis extension and a properly configured
 * Redis server.
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
    protected $connector;

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
     * @param int $db
     * @param string|null $password
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @param float $timeout
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(
        string $key,
        string $host = '127.0.0.1',
        int $port = 6379,
        int $db = 0,
        ?string $password = null,
        ?Serializer $serializer = null,
        float $timeout = 2.5
    ) {
        if (!extension_loaded('redis')) {
            throw new RuntimeException('Redis extension is not loaded, check'
                . ' PHP configuration.');
        }

        if (empty($key)) {
            throw new InvalidArgumentException('Key cannot be an empty string.');
        }

        if ($timeout <= 0.0) {
            throw new InvalidArgumentException('Timeout must be greater than'
                . " 0, $timeout given.");
        }

        $connector = new Redis();

        if (!$connector->connect($host, $port, $timeout)) {
            throw new RuntimeException('Could not connect to Redis server'
                . " at host $host on port $port.");
        }

        if (isset($password)) {
            if (!$connector->auth($password)) {
                throw new RuntimeException('Password is invalid.');
            }
        }

        if (!$connector->select($db)) {
            throw new RuntimeException("Failed to select database $db.");
        }

        $this->key = $key;
        $this->connector = $connector;
        $this->serializer = $serializer ?? new Native();
    }

    /**
     * Save the persistable object.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \RuntimeException
     */
    public function save(Persistable $persistable) : void
    {
        $data = $this->serializer->serialize($persistable);

        $success = $this->connector->set($this->key, $data);

        if (!$success) {
            throw new RuntimeException('Failed to save '
                . ' persistable to the database.');
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
        $data = $this->connector->get($this->key) ?: '';

        $persistable = $this->serializer->unserialize($data);

        return $persistable;
    }
}
