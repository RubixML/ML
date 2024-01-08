<?php

namespace Rubix\ML\Backends\Swoole;

class PhpSerializer implements Serializer
{
    public function serialize(mixed $value) : string
    {
        return serialize($value);
    }

    public function unserialize(string $data) : mixed
    {
        return unserialize($data);
    }
}
