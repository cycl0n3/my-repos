import React from 'react';

const AdminBlock = ({ user, children }) => {
  if (user.roles === 'ADMIN') {
    return (
      <>
        {children}
      </>
    );
  } else {
    return null;
  }
}

export default AdminBlock;
