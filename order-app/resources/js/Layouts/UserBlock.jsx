import React from 'react';

const UserBlock = ({user, children}) => {
  if (user.roles === 'USER') {
    return (
      <>
        {children}
      </>
    );
  } else {
    return null;
  }
};

export default UserBlock;
